import torch
import torch.nn as nn
import torch.nn.functional as F
from model.HolisticAttention import HA

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB(nn.Module):
    # RFB-like multi-scale module
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation model, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x



class SCA(nn.Module):
    def __init__(self):
        super(SCA, self).__init__()

        self.squeeze_rgb = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_rgb = nn.Sequential(
            nn.Conv2d(32, 32, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())


        self.squeeze_depth = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_depth = nn.Sequential(
            nn.Conv2d(32, 32, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())

        self.cross_conv = nn.Conv2d(32*2, 32, 1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x3_r,x3_d):
        SCA_ca = self.channel_attention_rgb(self.squeeze_rgb(x3_r))
        SCA_3_o = x3_r * SCA_ca.expand_as(x3_r)

        SCA_d_ca = self.channel_attention_depth(self.squeeze_depth(x3_d))
        SCA_3d_o = x3_d * SCA_d_ca.expand_as(x3_d)

        Co_ca3 = torch.softmax(SCA_ca + SCA_d_ca,dim=1)

        SCA_3_co = x3_r * Co_ca3.expand_as(x3_r)
        SCA_3d_co= x3_d * Co_ca3.expand_as(x3_d)

        CR_fea3_rgb = SCA_3_o + SCA_3_co
        CR_fea3_d = SCA_3d_o + SCA_3d_co

        CR_fea3 = torch.cat([CR_fea3_rgb,CR_fea3_d],dim=1)
        CR_fea3 = self.cross_conv(CR_fea3)

        return CR_fea3



class fusion(nn.Module):
    def __init__(self, in_channel=32, out_channel=32):
        super(fusion, self).__init__()

        channel = in_channel
        self.rfb3_1 = RFB(channel, channel)
        self.rfb4_1 = RFB(channel, channel)
        self.rfb5_1 = RFB(channel, channel)
        self.agg1 = aggregation(channel)

        self.rfb3_2 = RFB(channel, channel)
        self.rfb4_2 = RFB(channel, channel)
        self.rfb5_2 = RFB(channel, channel)
        self.agg2 = aggregation(channel)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        self.HA = HA()

        self.SCA3 = SCA()
        self.SCA4 = SCA()
        self.SCA5 = SCA()

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

        self._init_weight()



    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x3_r,x4_r,x5_r,x3_d,x4_d,x5_d):

        # Cross Reference Module
        x3 = self.SCA3(x3_r, x3_d)          # b_size,32, 1/8.  1/8    (44, 44)
        x4 = self.SCA4(x4_r, x4_d)          # b_size,32, 1/16. 1/16   (22, 22)
        x5 = self.SCA5(x5_r, x5_d)          # b_size,32, 1/32. 1/32   (11, 11)


        # Decoder
        x3_1 = self.rfb3_1(x3)
        x4_1 = self.rfb4_1(x4)
        x5_1 = self.rfb5_1(x5)
        attention_map = self.agg1(x5_1, x4_1, x3_1)
        x3_2 = self.HA(attention_map.sigmoid(), x3)
        x4_2 = self.conv4(x3_2)
        x5_2 = self.conv5(x4_2)
        x3_2 = self.rfb3_2(x3_2)
        x4_2 = self.rfb4_2(x4_2)
        x5_2 = self.rfb5_2(x5_2)
        detection_map = self.agg2(x5_2, x4_2, x3_2)

        return self.upsample(attention_map), self.upsample(detection_map), self.up8(x3), self.up16(x4), self.up32(x5)

