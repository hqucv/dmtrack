import torch.nn as nn

from mmcv.cnn import normal_init

__all__ = ['Cond_Conv']


class Cond_Conv(nn.Module):

    def __init__(self, feat_channels=256, modulator_channel=8, filter_size=1, relative_coords=True):
        super(Cond_Conv, self).__init__()
        if relative_coords:
            self.controller = nn.Conv2d(
                feat_channels,
                ((feat_channels+2)*filter_size*filter_size*modulator_channel
                + feat_channels*filter_size*filter_size*modulator_channel
                + feat_channels
                + modulator_channel)*2,
                kernel_size=1,
                stride=1, padding=1
            )
        else:
            self.controller_cls1 = nn.Conv2d(
                feat_channels,
                feat_channels*feat_channels,
                kernel_size=1,
                stride=1, padding=1
            )
            self.controller_cls2 = nn.Conv2d(
                feat_channels,
                modulator_channel*modulator_channel,
                kernel_size=1,
                stride=1, padding=1
            )
            self.controller_reg1 = nn.Conv2d(
                feat_channels,
                feat_channels*feat_channels,
                kernel_size=1,
                stride=1, padding=1
            )
            self.controller_reg2 = nn.Conv2d(
                feat_channels,
                modulator_channel*modulator_channel,
                kernel_size=1,
                stride=1, padding=1
            )
            self.out = nn.AdaptiveAvgPool2d(1)

    def forward(self, feat):
        return self.out(self.controller_cls1(feat)), \
               self.out(self.controller_cls2(feat)), \
               self.out(self.controller_reg1(feat)), \
               self.out(self.controller_reg2(feat))

    def init_weights(self):
        normal_init(self.controller_cls1, std=0.01)
        normal_init(self.controller_cls2, std=0.01)
        normal_init(self.controller_reg1, std=0.01)
        normal_init(self.controller_reg2, std=0.01)


