
# ---------------------------------------------------------
# GAN model with generator and discriminator !
# editor 2021 11 10
# ---------------------------------------------------------

import sys
sys.path.append('../')
sys.path.append('../networks/')
import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
from RCS_UNets import RCSU_Net

class standard_conv(nn.Module):
    def __init__(self, in_ch, out_ch, is_pooling=True, stride=1):
        super(standard_conv, self).__init__()
        if is_pooling is True:
            self.conv_layer = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
        else:
            self.conv_layer = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            )

    def forward(self, x):
        return self.conv_layer(x)

class standard_up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(standard_up, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = standard_conv(in_ch, out_ch, is_pooling=False)

    def forward(self, x_d, x_e):
        merge = torch.cat([self.up(x_d), x_e], dim=1)
        de_x = self.conv(merge)
        return de_x

class result_map(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(result_map, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        map = self.conv(x)
        map = self.activate(map)
        return map

class resnet_tail(nn.Module):
    def __init__(self, in_ch):
        super(resnet_tail, self).__init__()
        self.global_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(in_ch, 1)

    def forward(self, x):
        # output layer: (N, 16, 16, 512) -> (N, 1, 1, 512) -> (N, 1)
        new_x = self.global_pooling(x)
        new_x = new_x.reshape((-1, new_x.size()[1], 1)).permute((0, 2, 1))
        new_x = self.fc(new_x)
        return new_x.reshape((-1, 1))

class generator1(nn.Module):
    def __init__(self):
        super(generator1, self).__init__()
        filter_list = [1, 32, 64, 128, 256, 512]
        self.conv1 = standard_conv(filter_list[0], filter_list[1])
        self.conv2 = standard_conv(filter_list[1], filter_list[2])
        self.conv3 = standard_conv(filter_list[2], filter_list[3])
        self.conv4 = standard_conv(filter_list[3], filter_list[4])
        self.conv5 = standard_conv(filter_list[4], filter_list[5], is_pooling=False)

        self.up1 = standard_up(filter_list[5] + filter_list[3], filter_list[4])
        self.up2 = standard_up(filter_list[4] + filter_list[2], filter_list[3])
        self.up3 = standard_up(filter_list[3] + filter_list[1], filter_list[2])
        self.up4 = standard_up(filter_list[2] + filter_list[0], filter_list[1])
        self.result = result_map(32, 1)

    def forward(self, data):
        # conv1: (N, 512, 512, 3) -> (N, 256, 256, 32)
        conv_x1 = self.conv1(data)
        # conv2: (N, 256, 256, 32) -> (N, 128, 128, 64)
        conv_x2 = self.conv2(conv_x1)
        # conv3: (N, 128, 128, 64) -> (N, 64, 64, 128)
        conv_x3 = self.conv3(conv_x2)
        # conv4: (N, 64, 64, 128) -> (N, 32, 32, 256)
        conv_x4 = self.conv4(conv_x3)
        # conv5: (N, 32, 32, 256) -> (N, 16, 16, 512)
        conv_x5 = self.conv5(conv_x4)
        # conv6: (N, 16, 16, 512) -> (N, 32, 32, 256)
        conv6 = self.up1(conv_x5, conv_x3)
        # conv7: (N, 32, 32, 256) -> (N, 64, 64, 128)
        conv7 = self.up2(conv6, conv_x2)
        # conv8: (N, 64, 64, 128) -> (N, 128, 128, 64)
        conv8 = self.up3(conv7, conv_x1)
        # conv9: (N, 128, 128, 64) -> (N, 256, 256, 32)
        conv9 = self.up4(conv8, data)
        # output layer: (N, 256, 256, 32) -> (N, 512, 512, 1)
        conv10 = self.result(conv9)
        return conv10

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.nets_g = RCSU_Net(1, 1)

    def forward(self, data):
        output = self.nets_g(data)
        return output

class discriminator(nn.Module):
    def __init__(self, flags = 'image'):
        super(discriminator, self).__init__()
        self.flags = flags
        filters_list = [2, 32, 64, 128, 256, 512]
        self.conv1 = standard_conv(filters_list[0], filters_list[1], stride=2)
        self.conv2 = standard_conv(filters_list[1], filters_list[2], stride=2)
        self.conv3 = standard_conv(filters_list[2], filters_list[3], stride=1)
        self.conv4 = standard_conv(filters_list[3], filters_list[4], stride=1)
        self.conv5 = standard_conv(filters_list[4], filters_list[5], stride=1, is_pooling = False)
        self.tail  = resnet_tail(filters_list[5])
        self.acti  = nn.Sigmoid()
        # self.conv_x1 = standard_conv(filters_list[0], filters_list[1], stride=1, is_pooling=False)
        # self.conv_x2 = standard_conv(filters_list[1], filters_list[2], stride=1, is_pooling=False)
        # self.conv_x3 = standard_conv(filters_list[2], filters_list[3], stride=1, is_pooling=False)
        # self.output = standard_conv(filters_list[3], 1,  stride=1, is_pooling=False)

    def forward(self, data, is_reuse=False):
        if self.flags == 'pixel':
            return self.discriminator_pixel(data, is_reuse=is_reuse)
        elif self.flags == 'patch1':
            return self.discriminator_patch1(data, is_reuse=is_reuse)
        elif self.flags == 'patch2':
            return self.discriminator_patch2(data, is_reuse=is_reuse)
        elif self.flags == 'image':
            return self.discriminator_image(data, is_reuse=is_reuse)
        else:
            raise NotImplementedError

    # def discriminator_pixel(self, data, is_reuse=False):
    #     filters_list = [2, 32, 64, 128]
    #     # conv1: (N, 512, 512, 4) -> (N,, 512, 512, 32)
    #     conv_x11 = self.conv_x1(data)
    #     # conv2: (N, 512, 512, 32) -> (N, 512, 512, 64)
    #     conv_x21 = self.conv_x2(conv_x11)
    #     # conv3: (N, 512, 512, 64) -> (N, 512, 512, 128)
    #     conv_x31 = self.conv_x3(conv_x21)
    #     # output layer: (N, 512, 512, 128) -> (N, 512, 512, 1)
    #     output = self.output(conv_x31)
    #     return nn.Sigmoid()(output)

    # def discriminator_patch2(self, data, is_reuse=False):
    #     filters_list = [4, 32, 64, 128]
    #     # conv1: (N, 512, 512, 4) -> (N, 256, 256, 32)
    #     conv1 = standard_conv(filters_list[0], filters_list[1],  stride=2)(data)
    #     # conv2: (N, 256, 256, 32) -> (N, 128, 128, 64)
    #     conv2 = standard_conv(filters_list[1], filters_list[2],  stride=1)(conv1)
    #     # conv3: (N, 128, 128, 64) -> (N, 64, 64, 128)
    #     conv3 = standard_conv(filters_list[2], filters_list[3], stride=1)(conv2)
    #     # output layer: (N, 64, 64, 128) -> (N, 32, 32, 1)
    #     output = standard_conv(filters_list[3], 1, stride=1, is_pooling=False)(conv3)
    #     return  nn.Sigmoid()(output), output

    # def discriminator_patch1(self, data,  is_reuse=False):
    #     filters_list = [4, 32, 64, 128, 256, 512]
    #     # conv1: (N, 512, 512, 4) -> (N, 256, 256, 32)
    #     conv_x1 = standard_conv(filters_list[0], filters_list[1], stride=2)(data)
    #     # conv2: (N, 256, 256, 32) -> (N, 64, 64, 64)
    #     conv_x2 = standard_conv(filters_list[1], filters_list[2], stride=2)(conv_x1)
    #     # conv3: (N, 64, 64, 64) -> (N, 32, 32, 128)
    #     conv_x3 = standard_conv(filters_list[2], filters_list[3], stride=1)(conv_x2)
    #     # conv4: (N, 32, 32, 128) -> (N, 16, 16, 256)
    #     conv_x4 = standard_conv(filters_list[3], filters_list[4], stride=1)(conv_x3)
    #     # conv5: (N, 16, 16, 256) -> (N, 8, 8, 512)
    #     conv_x5 = standard_conv(filters_list[4], filters_list[5], stride=1, is_pooling=False)(conv_x4)
    #     # output layer: (N, 8, 8, 512) -> (N, 4, 4, 1)
    #     output = standard_conv(filters_list[4], filters_list[5], stride=1, is_pooling=False)(conv_x5)
    #     return nn.Sigmoid()(output), output

    def discriminator_image(self, data, is_reuse):
        # conv1: (N, 512, 512, 2) -> (N,, 256, 256, 32)
        conv_x1 = self.conv1(data)
        # conv2: (N, 256, 256, 32)-> (N, 128, 128, 64)
        conv_x2 = self.conv2(conv_x1)
        # conv3: (N, 128, 128, 64)-> (N, 64, 64, 128)
        conv_x3 = self.conv3(conv_x2)
        # conv4: (N, 64, 64, 128) -> (N, 32, 32, 256)
        conv_x4 = self.conv4(conv_x3)
        # conv5: (N, 32, 32, 256) -> (N, 16, 16, 512)
        conv_x5 = self.conv5(conv_x4)
        # output layer: (N, 16, 16, 512) -> (N, 1, 1, 512) -> (N, 1)
        output = self.tail(conv_x5)
        # print('Before activate value is:', output)
        outs = self.acti(output)
        # print('After activate value is:', outs)
        return outs

class Discriminator(nn.Module):
    def __init__(self, input_channels = 2, n_filters = 16, n_classes = 1, bilinear=False):
        super(Discriminator, self).__init__()
        print('new discriminator')
        self.n_channels = input_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(input_channels, n_filters)
        self.down1 = Down(n_filters, 2 * n_filters)
        self.down2 = Down(2 * n_filters, 4 * n_filters)
        self.down3 = Down(4 * n_filters, 8 * n_filters)
        self.down4 = Down(8 * n_filters, 16 * n_filters)

        self.up1 = Up(16 * n_filters, 8 * n_filters, bilinear)
        self.up2 = Up(8 * n_filters, 4 * n_filters, bilinear)
        self.up3 = Up(4 * n_filters, 2 * n_filters, bilinear)
        self.up4 = Up(2 * n_filters, 1 * n_filters, bilinear)
        self.outc = OutConv(n_filters, n_classes)

        self.acti = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return self.acti(logits)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv,self).__init__()
        mid_channels = out_channels
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)

        else:
            self.conv_bottom = bottom_conv(in_channels, out_channels)
            self.up = nn.Upsample(scale_factor=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.conv_bottom(x1)
        x1 = self.up(x1)
        # input size is [C H W] !
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class bottom_conv(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(bottom_conv, self).__init__()
        mid_channels = out_channels
        if not mid_channels:
            mid_channels = out_channels
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

if __name__ == '__main__':

    '''
    test generator and discriminator !
    '''

    original_img = torch.rand(size=(2, 1, 896, 896))
    out_seg = generator()(original_img)
    print(out_seg.size(), '--------------------')

    dis_img = torch.rand(size=(2 ,2, 896, 896))
    res_seg = discriminator()(dis_img)
    print(res_seg.size())
    res_seg2 = Discriminator()(dis_img)
    print(res_seg2.size())

    print()