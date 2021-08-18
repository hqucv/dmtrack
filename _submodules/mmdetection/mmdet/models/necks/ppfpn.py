import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from mmdet.core import auto_fp16
from ..builder import NECKS


@NECKS.register_module()
class PPFPN(nn.Module):
    """
    Feature Pyramid Network.

    This is an implementation of - Feature Pyramid Networks for Object
    Detection (https://arxiv.org/abs/1612.03144)

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(PPFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

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
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.upsample_convs = nn.ModuleList()
        self.out_convs = nn.ModuleList()

        for i in range(self.backbone_end_level-1, self.start_level-1, -1):
            if i == self.backbone_end_level-1:
                l_conv = ConvModule(
                    in_channels[i] + 2,
                    in_channels[i],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False)
            else:
                l_conv = ConvModule(
                    in_channels[i] * 2 + 2,
                    in_channels[i],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False)

            if i == self.start_level:
                u_conv = None
            else:
                u_conv = ConvModule(
                    in_channels[i] + 2,
                    int(in_channels[i] / 2),
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False)

            fpn_conv = nn.ModuleList([
                ConvModule(
                    in_channels[i],
                    in_channels[i] * 2,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False
                ),
                ConvModule(
                    in_channels[i] * 2 + 2,
                    in_channels[i],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False
                ),
                ConvModule(
                    in_channels[i],
                    in_channels[i] * 2,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False
                ),
                ConvModule(
                    in_channels[i] * 2 + 2,
                    in_channels[i],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False
                )
            ])

            out_conv = ConvModule(
                    in_channels[i],
                    self.out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.upsample_convs.append(u_conv)
            self.out_convs.append(out_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def concat_coord(self, input):
        # concat coord
        x_range = torch.linspace(-1, 1, input.shape[-1], device=input.device)
        y_range = torch.linspace(-1, 1, input.shape[-2], device=input.device)
        y, x = torch.meshgrid(y_range, x_range)

        y = y.expand([input.shape[0], 1, -1, -1])
        x = x.expand([input.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)

        input = torch.cat([input, coord_feat], 1)
        return input

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        inputs = inputs[::-1]
        out_feat = []
        for i in range(self.start_level, self.backbone_end_level):
            lateral_feat = self.concat_coord(inputs[i])
            lateral_feat = self.lateral_convs[i](lateral_feat)
            for idx, block in enumerate(self.fpn_convs[i]):
                if idx % 2 == 1:
                    lateral_feat = self.concat_coord(lateral_feat)
                lateral_feat = block(lateral_feat)
            out_feat.append(self.out_convs[i](lateral_feat))
            if i != self.backbone_end_level - 1:
                upsample_feat = self.concat_coord(lateral_feat)
                upsample_feat = self.upsample_convs[i](upsample_feat)
                if 'scale_factor' in self.upsample_cfg:
                    upsample_feat = F.interpolate(upsample_feat, **self.upsample_cfg)
                else:
                    prev_shape = inputs[i + 1].shape[2:]
                    upsample_feat = F.interpolate(upsample_feat, size=prev_shape, **self.upsample_cfg)
                inputs[i+1] = torch.cat([inputs[i+1], upsample_feat], 1)
        out_feat = out_feat[::-1]
        return tuple(out_feat)

