import torch
import torch.nn as nn
from collections import OrderedDict


def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, groups=groups,bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU6(inplace=True)),
    ]))

#-----------------------------------
#   深度可分离卷积DSC=DW+PW
#-----------------------------------
def conv_dw(filter_in, filter_out, stride=1):
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_in, 3, stride, 1, groups=filter_in, bias=False),
        nn.BatchNorm2d(filter_in),
        nn.ReLU6(inplace=True),

        nn.Conv2d(filter_in, filter_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.ReLU6(inplace=True),
    )


# ---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化固定输出大小 1，3，5，9
# ---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features


class Upsample(nn.Module):
    def __init__(self, filter_in, filter_out):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            # conv2d(in_channels, out_channels, 1),
            conv_dw(filter_in, filter_out, stride=1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x

# 我们的卷积块采用DSC与Conv交替的方式，即可保证精度也可减少计算，起到了平衡
# ---------------------------------------------------#
#   三次卷积块
# ---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


# ---------------------------------------------------#
#   五次卷积块
# ---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv_dw(in_filters, filters_list[0]),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


class LightweightFeaturePyramidNet(nn.Module):
    def __init__(self):
        super(LightweightFeaturePyramidNet, self).__init__()
        in_filters = [40, 112, 160] #为了降维，减少计算量，加卷积层后的滤镜通道从40改到160

        self.fpn_conv1 = make_three_conv([512, 1024], in_filters[0])
        self.fpn_SPP = SpatialPyramidPooling()
        self.fpn_conv2 = make_three_conv([512, 1024], 2048)

        self.fpn_upsample1 = Upsample(512, 256)
        self.fpn_make_five_conv1 = make_five_conv([256, 512], 256)

        self.fpn_upsample2 = Upsample(256, 128)
        self.fpn_make_five_conv2 = make_five_conv([128, 256], 128)

        self.fpn_down_sample1 = conv_dw(128, 256, stride=2)
        self.fpn_make_five_conv3 = make_five_conv([256, 512], 512)

        self.fpn_down_sample2 = conv_dw(256, 512, stride=2)
        self.fpn_make_five_conv4 = make_five_conv([512, 1024], 1024)

    def forward(self, x):
        #  backbone 主干网络的维度变化
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048
        P5 = self.lfpn_conv1(x)
        P5 = self.lfpn_SPP(P5)  #经过SPP后将输入特征映射固定尺寸大小，这样的方式和resize有区别
        # 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.lfpn_conv2(P5)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        P5_upsample = self.lfpn_upsample1(P5)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.lfpn_make_five_conv1(P5_upsample)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        P4_upsample = self.lfpn_upsample2(P4)
        # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        P3 = self.lfpn_make_five_conv2(P4_upsample)

        # 52,52,128 -> 26,26,256
        P3_downsample = self.lfpn_down_sample1(P3)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P3_downsample, P4], axis=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.lfpn_make_five_conv3(P4)

        # 26,26,256 -> 13,13,512
        P4_downsample = self.lfpn_down_sample2(P4)
        # 13,13,512 + 13,13,512 -> 13,13,1024
        P5 = torch.cat([P4_downsample, P5], axis=1)
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.lfpn_make_five_conv4(P5)

        return P5
