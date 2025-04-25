import os
import torch
import torch.nn as nn


class HyNetSingleLevel(nn.Module):
    def __init__(self, msi_c=3, hsi_c=31, nf=64, res_number=10):
        super(HyNetSingleLevel, self).__init__()
        self.res_number = res_number

        # Deconvolution layer
        self.deconv_feature = nn.ConvTranspose2d(nf, nf, kernel_size=6, stride=2, padding=2, output_padding=0)
        self.concat_conv1 = nn.Conv2d(msi_c + nf, msi_c + nf, kernel_size=3, stride=1, padding=1)
        self.concat_conv2 = nn.Conv2d(msi_c + nf, nf, kernel_size=3, stride=1, padding=1)

        # Residual blocks      # nn.ModuleList
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
            ) for _ in range(res_number)
        ])

        # Gradient prediction layer
        self.gradient_level = nn.Conv2d(nf, hsi_c, kernel_size=3, stride=1, padding=1)

        # Upsampling layer
        # 先上采样后进行像素洗牌
        self.upconv_image = nn.Conv2d(hsi_c, hsi_c * 4, kernel_size=3, stride=1, padding=1)
        self.subpixel_image = nn.PixelShuffle(2)

    def forward(self, net_image, net_feature, rgbDownsample):
        net_feature = self.deconv_feature(net_feature)
        concat_feature = torch.cat([net_feature, rgbDownsample], dim=1)
        net_feature = self.concat_conv1(concat_feature)
        net_feature = self.concat_conv2(net_feature)

        # Residual blocks
        for block in self.residual_blocks:
            net_tmp = block(net_feature)
            net_feature = net_feature + net_tmp

        # Gradient prediction
        # 将得到的Resnet特征图通道数恢复到HSI_bands
        gradient_level = self.gradient_level(net_feature)

        # Upsampling
        net_image = self.upconv_image(net_image)
        net_image = self.subpixel_image(net_image)

        # Add gradient to image
        net_image = net_image + gradient_level

        return net_image, net_feature


class HyTestNet(nn.Module):
    def __init__(self, msi_c=3, hsi_c=31, nf=64, res_number=10):
        super(HyTestNet, self).__init__()
        self.res_number = res_number

        # Initial convolution layers
        self.init_conv = nn.Conv2d(hsi_c, nf, kernel_size=3, stride=1, padding=1)
        self.init_conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)

        # RGB downsampling layers
        self.rgbdown1 = nn.Conv2d(msi_c, msi_c, kernel_size=6, stride=2, padding=2)
        self.rgbdown2 = nn.Conv2d(msi_c, msi_c, kernel_size=6, stride=2, padding=2)

        # Single level networks
        self.level1 = HyNetSingleLevel(msi_c, hsi_c, nf, res_number)
        self.level2 = HyNetSingleLevel(msi_c, hsi_c, nf, res_number)
        self.level3 = HyNetSingleLevel(msi_c, hsi_c, nf, res_number)

    def forward(self, inputs, rgb_image):
        net_feature = self.init_conv(inputs)
        net_feature = self.init_conv2(net_feature)

        # RGB downsampling
        rgbSample1 = self.rgbdown1(rgb_image)
        rgbSample2 = self.rgbdown2(rgbSample1)

        # Single level networks
        net_image1, net_feature1 = self.level1(inputs, net_feature, rgbSample2)
        net_image2, net_feature2 = self.level2(net_image1, net_feature1, rgbSample1)
        net_image3, net_feature3 = self.level3(net_image2, net_feature2, rgb_image)

        return net_image3, net_image2, net_image1


if __name__ == '__main__':
    model = HyTestNet()
    Y = torch.rand(1, 31, 10, 10)
    Z = torch.rand(1, 3, 80, 80)
    x3, x2, x1 = model(Y, Z)
    print(x3.shape, x2.shape, x1.shape)
