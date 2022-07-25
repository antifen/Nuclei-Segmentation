import torch
import torch.nn as nn
import torch.optim as optimizer
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
        # print(x_d.size(), x_e.size(),'-------')
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
        # output layer: (N, 8, 8, 512) -> (N, 1, 1, 512) -> (N, 1)
        new_x = self.global_pooling(x)
        new_x = new_x.reshape((-1, new_x.size()[1], 1)).permute((0, 2, 1))
        new_x = self.fc(new_x)
        return new_x.reshape((-1, 1))

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = RCSU_Net(3, 1)

    def forward(self, data):
        x = self.gen(data)
        return x

class discriminator(nn.Module):

    def __init__(self, flags='image'):
        super(discriminator, self).__init__()
        self.flags = flags
        filters_list = [4, 32, 64, 128, 256, 512]
        self.conv1 = standard_conv(filters_list[0], filters_list[1], stride=2)
        self.conv2 = standard_conv(filters_list[1], filters_list[2], stride=2)
        self.conv3 = standard_conv(filters_list[2], filters_list[3], stride=1)
        self.conv4 = standard_conv(filters_list[3], filters_list[4], stride=1)
        self.conv5 = standard_conv(filters_list[4], filters_list[5], stride=1, is_pooling=False)
        self.tail = resnet_tail(filters_list[5])
        self.acti = nn.Sigmoid()

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

    def discriminator_pixel(self, data, is_reuse=False):
        filters_list = [4, 32, 64, 128]
        # conv1: (N, 512, 512, 4) -> (N,, 512, 512, 32)
        conv_x1 = standard_conv(filters_list[0], filters_list[1], stride=1, is_pooling=False)(data)
        # conv2: (N, 512, 512, 32) -> (N, 512, 512, 64)
        conv_x2 = standard_conv(filters_list[1], filters_list[2], stride=1, is_pooling=False)(conv_x1)
        # conv3: (N, 512, 512, 64) -> (N, 512, 512, 128)
        conv_x3 = standard_conv(filters_list[2], filters_list[3], stride=1, is_pooling=False)(conv_x2)
        # output layer: (N, 512, 512, 128) -> (N, 512, 512, 1)
        output = standard_conv(filters_list[3], filters_list[0],  stride=1, is_pooling=False)(conv_x3)
        return nn.Sigmoid()(output), output

    def discriminator_patch2(self, data, is_reuse=False):
        filters_list = [4, 32, 64, 128]
        # conv1: (N, 512, 512, 4) -> (N,, 512, 512, 32)
        conv1 = standard_conv(filters_list[0], filters_list[1],  stride=2)(data)
        # conv2: (N, 256, 256, 32) -> (N, 128, 128, 64)
        conv2 = standard_conv(filters_list[1], filters_list[2],  stride=1)(conv1)
        # conv3: (N, 128, 128, 64) -> (N, 64, 64, 128)
        conv3 = standard_conv(filters_list[2], filters_list[3], stride=1)(conv2)
        # output layer: (N, 64, 64, 128) -> (N, 32, 32, 1)
        output = standard_conv(filters_list[3], 1, stride=1, is_pooling=False)(conv3)
        return  nn.Sigmoid()(output), output

    def discriminator_patch1(self, data,  is_reuse=False):
        filters_list = [4, 32, 64, 128, 256, 512]
        # conv1: (N, 512, 512, 4) -> (N,, 256, 256, 32)
        conv_x1 = standard_conv(filters_list[0], filters_list[1], stride=2)(data)
        # conv2: (N, 256, 256, 32) -> (N, 128, 128, 64)
        conv_x2 = standard_conv(filters_list[1], filters_list[2], stride=2)(conv_x1)
        # conv3: (N, 128, 128, 64) -> (N, 64, 64, 128)
        conv_x3 = standard_conv(filters_list[2], filters_list[3], stride=1)(conv_x2)
        # conv4: (N, 64, 64, 128) -> (N, 32, 32, 256)
        conv_x4 = standard_conv(filters_list[3], filters_list[4], stride=1)(conv_x3)
        # conv5: (N, 32, 32, 256) -> (N, 16, 16, 512)
        conv_x5 = standard_conv(filters_list[4], filters_list[5], stride=1, is_pooling=False)(conv_x4)
        # output layer: (N, 16, 16, 512) -> (N, 8, 8, 1)
        output = standard_conv(filters_list[4], filters_list[5], stride=1, is_pooling=False)(conv_x5)
        return nn.Sigmoid()(output), output

    def discriminator_image(self, data, is_reuse):
        # conv1: (N, 512, 512, 4)  -> (N, 256, 256, 32)
        conv_x1 = self.conv1(data)
        # conv2: (N, 256, 256, 32) -> (N, 128, 128, 64)
        conv_x2 = self.conv2(conv_x1)
        # conv3: (N, 128, 128, 64) -> (N, 64, 64, 128)
        conv_x3 = self.conv3(conv_x2)
        # conv4: (N, 64, 64, 128) -> (N, 32, 32, 256)
        conv_x4 = self.conv4(conv_x3)
        # conv5: (N, 32, 32, 256) -> (N, 16, 16, 512)
        conv_x5 = self.conv5(conv_x4)
        # output layer: (N, 16, 16, 512) -> (N, 1, 1, 512) -> (N, 1)
        output = self.tail(conv_x5)
        # print('before activate value is:', output)
        outs = self.acti(output)
        # print('after activate value is:', outs)
        return outs

if __name__ == '__main__':
    '''
    test generator and discriminator 
    '''
    original_img = torch.rand(size=(2, 3, 128, 128))
    out_seg = generator()(original_img)
    print('Output size check: ', out_seg.size())
    dis_seg = discriminator()(torch.rand(size=(2, 4, 128, 128)))
    print('Output size check: ', dis_seg.size())
    # dis_img = torch.rand(size=(7,4,640,640))
    # _, res_seg = discriminator()(dis_img)
    # print(res_seg.size())

    print()