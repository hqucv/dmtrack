'''
    this is modified from https://github.com/toandaominh1997/EfficientDet.Pytorch.
    the implementation of extra layers would be not correct.

'''

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from ..builder import NECKS

import torch

eps = 0.0001


@NECKS.register_module
class BiFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 stack=1,
                 add_extra_convs=True,
                 extra_convs_on_inputs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=False),
                 act_cfg=dict(type='ReLU')):
        super(BiFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.act_cfg = act_cfg
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.stack = stack

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.stack_bifpn_convs = nn.ModuleList()

        self.extra_levels = num_outs - self.backbone_end_level + self.start_level

        if self.add_extra_convs:
            self.extra_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=None,
                inplace=False)
            self.lateral_convs.append(l_conv)

        if self.extra_levels > 0:
            for i in range(self.extra_levels):
                in_channels = self.in_channels[self.backbone_end_level - 1]
                extra_l_conv = ConvModule(
                    in_channels,
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=None,
                    inplace=False)
                self.lateral_convs.append(extra_l_conv)

                if self.add_extra_convs:
                    extra_conv = ConvModule(
                        in_channels,
                        in_channels,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=self.act_cfg,
                        inplace=False)
                    self.extra_convs.append(extra_conv)

        for ii in range(stack):
            self.stack_bifpn_convs.append(BiFPNModule(channels=out_channels,
                                                      levels=self.backbone_end_level - self.start_level + self.extra_levels,
                                                      conv_cfg=conv_cfg,
                                                      norm_cfg=norm_cfg,
                                                      act_cfg=act_cfg))

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = list(inputs)

        # add extra
        if self.extra_levels > 0:
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            for i in range(self.extra_levels):
                if self.add_extra_convs:
                    inputs.append(self.extra_convs[i](inputs[-1]))
                else:
                    inputs.append(F.max_pool2d(inputs[-1], 1, stride=2))

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # part 1: build top-down and down-top path with stack
        for bifpn_module in self.stack_bifpn_convs:
            laterals = bifpn_module(laterals)
        outs = laterals
        return tuple(outs)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


class BiFPNModule(nn.Module):
    def __init__(self,
                 channels,
                 levels,
                 init=0.5,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(BiFPNModule, self).__init__()
        self.act_cfg = act_cfg
        self.levels = levels
        self.bifpn_convs = nn.ModuleList()
        # weighted
        self.w1 = nn.Parameter(torch.Tensor(2, levels).fill_(init))
        self.relu1 = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, levels - 2).fill_(init))
        self.relu2 = nn.ReLU()
        for jj in range(2):
            for i in range(self.levels - 1):  # 1,2,3
                fpn_conv = nn.Sequential(
                    ConvModule(
                        channels,
                        channels,
                        3,
                        padding=1,
                        groups=channels,
                        conv_cfg=conv_cfg,
                        norm_cfg=None,
                        act_cfg=None,
                        inplace=False),
                    ConvModule(
                        channels,
                        channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=self.act_cfg,
                        inplace=False))
                self.bifpn_convs.append(fpn_conv)

    def forward(self, inputs):
        assert len(inputs) == self.levels
        # build top-down and down-top path with stack
        levels = self.levels
        # w relu
        w1 = self.relu1(self.w1)
        w1 /= torch.sum(w1, dim=0) + eps  # normalize
        w2 = self.relu2(self.w2)
        w2 /= torch.sum(w2, dim=0) + eps
        # build top-down
        kk = 0
        # pathtd = inputs copy is wrong
        pathtd = [inputs[levels - 1]]
        #        for in_tensor in inputs:
        #            pathtd.append(in_tensor.clone().detach())
        for i in range(levels - 1, 0, -1):
            _t = w1[0, kk] * inputs[i - 1] + w1[1, kk] * F.interpolate(
                pathtd[-1], scale_factor=2, mode='nearest')
            pathtd.append(self.bifpn_convs[kk](_t))
            del (_t)
            kk = kk + 1
        jj = kk
        pathtd = pathtd[::-1]
        # build down-top
        for i in range(0, levels - 2, 1):
            pathtd[i + 1] = w2[0, i] * inputs[i + 1] + w2[1, i] * nn.Upsample(scale_factor=0.5)(pathtd[i]) + w2[2, i] * \
                            pathtd[i + 1]
            pathtd[i + 1] = self.bifpn_convs[jj](pathtd[i + 1])
            jj = jj + 1

        pathtd[levels - 1] = w1[0, kk] * inputs[levels - 1] + w1[1, kk] * nn.Upsample(scale_factor=0.5)(
            pathtd[levels - 2])
        pathtd[levels - 1] = self.bifpn_convs[jj](pathtd[levels - 1])
        return pathtd

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
