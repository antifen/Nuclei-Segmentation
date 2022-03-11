import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
nonlinearity = partial(F.relu, inplace=True)
from recurrent_convolution import RRCNN_block
from SC_MT_attention import SCMT_Attn

class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, attn=False):
        super(ResidualBlock, self).__init__()
        self.fuse_attn = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.Softmax2d() if attn else nn.PReLU()
        )
        self.residual_link = nn.Conv2d(in_ch, out_ch, 1)
        # residual connections with 1*1

    def forward(self, x):
        # residual connections
        y = self.residual_link(x)+self.fuse_attn(x)
        return y

class DecoderBlock(nn.Module):
    def  __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        #  downsampling gating signal convolution
        g1 = self.W_g(g)
        # upsampling  l convolution
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel to 1 and use Sigmoid to get weight matrix
        psi = self.psi(psi)
        # return weighted  x
        return x * psi


class RCSU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=3):
        super(RCSU_Net, self).__init__()

        self.filters =[64//2, 128//2, 256//2, 512//2, 1024//2]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ResidualBlock(img_ch, self.filters[0])
        self.Conv2 = ResidualBlock(self.filters[0], self.filters[1])
        self.rcnn_unit2 = RRCNN_block(self.filters[1],self.filters[1])
        self.scmt2 = SCMT_Attn(in_channels=self.filters[1], out_channels=self.filters[1])

        self.Conv3 = ResidualBlock(self.filters[1], self.filters[2])
        self.rcnn_unit3 = RRCNN_block(self.filters[2],self.filters[2])
        self.scmt3 = SCMT_Attn(in_channels=self.filters[2], out_channels=self.filters[2])

        self.Conv4 = ResidualBlock(self.filters[2], self.filters[3])
        self.rcnn_unit4 = RRCNN_block(self.filters[3], self.filters[3])
        self.scmt4 = SCMT_Attn(in_channels=self.filters[3], out_channels=self.filters[3])

        self.Conv5 = ResidualBlock(self.filters[3], self.filters[4])

        self.Up5 = DecoderBlock(self.filters[4],self.filters[3])
        self.Att5 = Attention_block(F_g=self.filters[3], F_l=self.filters[3], F_int=self.filters[2])
        self.Up_conv5 = ResidualBlock(self.filters[4], self.filters[3])
        self.rcnn_unit5= RRCNN_block(self.filters[3], self.filters[3])
        self.scmt5 = SCMT_Attn(in_channels=self.filters[3], out_channels=self.filters[3])

        self.Up4 = DecoderBlock(self.filters[3], self.filters[2])
        self.Att4 = Attention_block(F_g=self.filters[2], F_l=self.filters[2], F_int=self.filters[1])
        self.Up_conv4 = ResidualBlock(self.filters[3], self.filters[2])
        self.rcnn_unit6= RRCNN_block(self.filters[2], self.filters[2])
        self.scmt6 = SCMT_Attn(in_channels=self.filters[2], out_channels=self.filters[2])

        self.Up3 = DecoderBlock(self.filters[2], self.filters[1])
        self.Att3 = Attention_block(F_g=self.filters[1], F_l=self.filters[1], F_int=self.filters[0])
        self.Up_conv3 = ResidualBlock(self.filters[2], self.filters[1])
        self.rcnn_unit7= RRCNN_block(self.filters[1], self.filters[1])
        self.scmt7 = SCMT_Attn(in_channels=self.filters[1], out_channels=self.filters[1])

        self.Up2 = DecoderBlock(self.filters[1], self.filters[0])
        self.Att2 = Attention_block(F_g=self.filters[0], F_l=self.filters[0], F_int=self.filters[0]//2)
        self.Up_conv2 = ResidualBlock(self.filters[1], self.filters[0])

        self.Conv_1x1 = ResidualBlock(self.filters[0], output_ch)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2 = self.rcnn_unit2(x2)
        x2 = self.scmt2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = self.rcnn_unit3(x3)
        x3 = self.scmt3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4 = self.rcnn_unit4(x4)
        x4 = self.scmt4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d5 = self.rcnn_unit5(d5)
        d5 = self.scmt5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d4 = self.rcnn_unit6(d4)
        d4 = self.scmt6(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d3 = self.rcnn_unit7(d3)
        d3 = self.scmt7(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

if __name__ == '__main__':
    d1 = RCSU_Net(3,1)(torch.rand(size=(5, 3, 128, 128)))
    print(d1.size(),'===========')
    pass