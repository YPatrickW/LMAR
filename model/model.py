try:
    from .module import *
except:
    from module import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class SuperUnet_MS(nn.Module):
    def __init__(self, channels, block="INV"):
        super(SuperUnet_MS, self).__init__()
        # ---------ENCODE
        self.layer_dowm1 = basic_block(channels, channels, block)
        self.dowm1 = nn.Sequential(nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=True),
                                   nn.InstanceNorm2d(channels * 2, affine=True), nn.LeakyReLU(0.2, inplace=True))
        self.layer_dowm2 = basic_block(channels * 2, channels * 2, block)
        self.dowm2 = nn.Sequential(nn.Conv2d(channels * 2, channels * 4, 4, 2, 1, bias=True),
                                   nn.InstanceNorm2d(channels * 4, affine=True), nn.LeakyReLU(0.2, inplace=True))
        # ---------DECODE
        self.layer_bottom = basic_block(channels * 4, channels * 4, block)
        self.up2 = nn.Sequential(nn.ConvTranspose2d(channels * 4, channels * 2, 4, 2, 1, bias=True),
                                 nn.InstanceNorm2d(channels * 2, affine=True), nn.LeakyReLU(0.2, inplace=True))
        self.layer_up2 = basic_block(channels * 2, channels * 2, block)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1, bias=True),
                                 nn.InstanceNorm2d(channels, affine=True), nn.LeakyReLU(0.2, inplace=True))
        self.layer_up1 = basic_block(channels, channels, block)
        # ---------SKIP
        self.fus2 = skip(channels * 4, channels * 2, "HIN")
        self.fus1 = skip(channels * 2, channels, "HIN")
        # ---------SKIP
        self.skip_down1 = nn.Sequential(nn.Conv2d(channels, channels, 4, 2, 1, bias=True),
                                        nn.InstanceNorm2d(channels, affine=True), nn.LeakyReLU(0.2, inplace=True))
        self.skip1 = skip(channels * 3, channels * 2, "CONV")
        self.skip_down2 = nn.Sequential(nn.Conv2d(channels * 2, channels, 4, 2, 1, bias=True),
                                        nn.InstanceNorm2d(channels, affine=True), nn.LeakyReLU(0.2, inplace=True))
        self.skip2 = skip(channels * 5, channels * 4, "CONV")
        # self.skip3 = skip(channels*2, channels, "CONV")
        self.skip_up4 = nn.Sequential(nn.ConvTranspose2d(channels * 4, channels, 4, 2, 1, bias=True),
                                      nn.InstanceNorm2d(channels, affine=True), nn.LeakyReLU(0.2, inplace=True))
        self.skip4 = skip(channels * 3, channels * 2, "CONV")
        # self.skip5 = skip(channels*2, channels, "CONV")
        self.skip_up6 = nn.Sequential(nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1, bias=True),
                                      nn.InstanceNorm2d(channels, affine=True), nn.LeakyReLU(0.2, inplace=True))
        self.skip6 = skip(channels * 2, channels, "CONV")

    def forward(self, x):
        # ---------ENCODE
        x_11 = self.layer_dowm1(x)
        x_down1 = self.dowm1(x_11)
        # x  =self.skip_down1(x)
        # print(x.shape, x_down1.shape)

        x_down1 = self.skip1(torch.cat([self.skip_down1(x), x_down1], 1), x_down1)

        x_12 = self.layer_dowm2(x_down1)
        x_down2 = self.dowm2(x_12)
        x_down2 = self.skip2(torch.cat([self.skip_down2(x_down1), x_down2], 1), x_down2)

        x_bottom = self.layer_bottom(x_down2)

        # ---------DECODE
        x_up2 = self.up2(x_bottom)
        x_22 = self.layer_up2(x_up2)
        x_22 = self.skip4(torch.cat([self.skip_up4(x_bottom), x_22], 1), x_22)
        x_22 = self.fus2(torch.cat([x_12, x_22], 1), x_22)

        x_up1 = self.up1(x_22)
        x_21 = self.layer_up1(x_up1)
        x_21 = self.skip6(torch.cat([self.skip_up6(x_22), x_21], 1), x_21)
        x_21 = self.fus1(torch.cat([x_11, x_21], 1), x_21)
        return x_21, x_down2


class skip(nn.Module):
    def __init__(self, channels_in, channels_out, block):
        super(skip, self).__init__()
        if block == "CONV":
            self.body = nn.Sequential(nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=True),
                                      nn.InstanceNorm2d(channels_out, affine=True), nn.ReLU(inplace=True), )
        if block == "ID":
            self.body = nn.Identity()
        if block == "INV":
            self.body = nn.Sequential(InvBlock(channels_in, channels_in // 2),
                                      nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=True), )
        if block == "HIN":
            self.body = nn.Sequential(HinBlock(channels_in, channels_out))
        # --------------------------------------
        self.alpha1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha1.data.fill_(1.0)
        self.alpha2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha2.data.fill_(0.5)

    def forward(self, x, y):
        out = self.alpha1 * self.body(x) + self.alpha2 * y
        return out


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'HIN':
            return HinBlock(channel_in, channel_out)

    return constructor



class InvBlock(nn.Module):
    def __init__(self, channel_num, channel_split_num, subnet_constructor=subnet('HIN'),
                 clamp=0.8):  ################  split_channel一般设为channel_num的一半
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out + x


class sample_block(nn.Module):
    def __init__(self, channels_in, channels_out, size, dil):
        super(sample_block, self).__init__()
        # ------------------------------------------
        if size == "DOWN":
            self.conv = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, 1, dil, dilation=dil),
                nn.InstanceNorm2d(channels_out, affine=True),
                nn.ReLU(inplace=True),
            )
        if size == "UP":
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(channels_in, channels_out, 3, 1, dil, dilation=dil),
                nn.InstanceNorm2d(channels_out, affine=True),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


class HinBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(HinBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_1 = nn.Sequential(nn.LeakyReLU(0.2, inplace=False), )
        self.conv_2 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.LeakyReLU(0.2, inplace=False), )

    def forward(self, x):
        out = self.conv_1(x)
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.conv_2(out)
        out += self.identity(x)
        return out
        
class basic_block(nn.Module):
    def __init__(self, channels_in, channels_out, block):
        super(basic_block, self).__init__()
        # ------------------------------------------
        if block == "CONV":
            self.body = nn.Sequential(nn.Conv2d(channels_in, channels_out, 3, 1, 1, bias=True),
                                      nn.InstanceNorm2d(channels_out, affine=True), nn.ReLU(inplace=True), )
        if block == "INV":
            self.body = nn.Sequential(InvBlock(channels_in, channels_out // 2))
        if block == "HIN":
            self.body = nn.Sequential(HinBlock(channels_in, channels_out))

        # self.use_frequency = use_frequency
        # if self.use_frequency:
        #     self.ff_branch = frequency_branch(channels_in, channels_out)
        #     self.skip = skip(channels_out*2, channels_out, "CONV")

    def forward(self, x):
        return self.body(x)

class net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args.model
        self.hr_inc = DoubleConv(self.args["in_channel"], self.args["model_channel"] * 2)
        self.hr_backbone = SuperUnet_MS(self.args["model_channel"] * 2)
        self.final_out = nn.Conv2d(self.args["model_channel"] * 2, 3, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.hr_inc(x)
        x, mid_feat = self.hr_backbone(x)
        out = self.final_out(x)
        return out

