import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary

class SE(nn.Module):
    def __init__(self, channel, r=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c , _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale
        y = torch.mul(x, y)
        return y


class SCMT_Attn(nn.Module):
    def __init__(self, in_channels, out_channels, need_spatial = False):
        super(SCMT_Attn,self).__init__()
        self.groups_cnn = 1         # you can change the groups !
        self.need_spatial = need_spatial
        self.conv_origin_s = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,  kernel_size=3, padding=1,
                                       stride= 1, groups = self.groups_cnn)
        self.conv_origin_c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,
                                     stride=1, groups = self.groups_cnn)
        self.conv_s = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.conv_c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.conv_to_origin =nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.activates = nn.Softmax(dim=2)
        # self.parameter = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        # self.pool1 = nn.MaxPool2d(3, padding=1, stride=2)
        # self.up1 = nn.ConvTranspose2d(in_channels=128*2, out_channels=128, kernel_size=2, stride=2)

    def channel_op(self,x):
        new_x = self.conv_c(x)   # N C H W
        # N C H*W
        new_x1 = torch.reshape(new_x, (new_x.size()[0], new_x.size()[1], new_x.size()[2] * new_x.size()[3]))
        # N H*W C
        new_x2 = new_x1.permute((0, 2, 1))
        # N C C
        mapping_channels = torch.bmm(new_x1, new_x2)
        mapping_channels = self.activates(mapping_channels)
        # N C H W
        target_cx = self.conv_origin_c(x)
        # N  C  H*W
        target_cx2 = torch.reshape(target_cx, (
        target_cx.size()[0], target_cx.size()[1], target_cx.size()[2] * target_cx.size()[3]))
        # N  H*W C   *   N  C  C
        bmm1 = target_cx2.permute((0, 2, 1))
        bmm2 = mapping_channels.permute((0, 2, 1))
        # N   H*W   C
        results_c1 = torch.bmm(bmm1, bmm2)
        # N   C  H*W
        results_c2 = results_c1.permute((0, 2, 1))
        # N C H W
        final_results_c = results_c2.reshape(
            (target_cx.size()[0], target_cx.size()[1], target_cx.size()[2], target_cx.size()[3]))
        # print('the final results of final is: ', final_results_c.size())

        return final_results_c

    def spatial_op(self, x):
        new_x = self.conv_s(x)  # N C H W
        # N C H*W
        new_x1 = torch.reshape(new_x, (new_x.size()[0], new_x.size()[1], new_x.size()[2] * new_x.size()[3]))
        # N H*W C
        new_x2 = new_x1.permute((0, 2, 1))
        # N H*W H*W
        mapping_channels = torch.bmm(new_x2, new_x1)
        mapping_channels = self.activates(mapping_channels)
        # N C H W
        target_sx = self.conv_origin_s(x)
        # N  C  H*W
        target_sx2 = torch.reshape(target_sx, (
            target_sx.size()[0], target_sx.size()[1], target_sx.size()[2] * target_sx.size()[3]))
        # N   C  H*W      *    N  H*W  H*W
        bmm1 = target_sx2
        bmm2 = mapping_channels.permute((0, 2, 1))
        # N   C  H*W
        results_s1 = torch.bmm(bmm1, bmm2)
        final_results_s = results_s1.reshape(
            (target_sx.size()[0], target_sx.size()[1], target_sx.size()[2], target_sx.size()[3]))
        # print('the final results of final is: ', final_results_s.size())

        return final_results_s

    def forward(self,x):
        '''
        two types of attentions are merged here to get results
        '''
        if self.need_spatial is True:
            channel_results = self.channel_op(x)
            spatial_results = self.spatial_op(x)
            s_c_att = torch.cat([channel_results, spatial_results], dim=1)
            s_c_results = self.conv_to_origin(s_c_att)
            return s_c_results
        else:
            channel_results = self.channel_op(x)
            return channel_results

def test_self_Att():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('current device is :', device)
    tensors = torch.randn(1, 32, 16, 16, dtype=torch.float)
    exam1 = SCMT_Attn(32, 'relu')
    exam1.to(device)
    for i in exam1.children():
        print(i)
    # summary(exam1, input_size=(32, 16, 16), batch_size=-1)
    # tensors = tensors.cuda()
    outputs = exam1(tensors)
    print(outputs[0].size())
    return

if __name__ == '__main__':
    new_tensor = torch.rand(size=(1,3,128,128))
    structures = SCMT_Attn(in_channels=3, out_channels=30, need_spatial = True)(new_tensor)
    print(structures.size())
    # summary(SCMT_Attn(in_channels=3, out_channels=30), input_size=(3, 32, 32), batch_size=-1)