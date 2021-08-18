import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

__all__ = ['Cond_Modulator']


def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


class Cond_Modulator(nn.Module):

    def __init__(self, feat_channel=256, modulator_channel=8, relative_coords=True):
        super(Cond_Modulator, self).__init__()
        self.relative_coords = relative_coords
        self.downsampler_cls = nn.Conv2d(
            feat_channel, modulator_channel, kernel_size=1,
            stride=1, padding=0
        )
        self.downsampler_reg = nn.Conv2d(
            feat_channel, modulator_channel, kernel_size=1,
            stride=1, padding=0
        )

    def forward(self, cls_x, reg_x, params):
        cls_x = self.downsampler_cls(cls_x)
        reg_x = self.downsampler_reg(reg_x)

        if self.relative_coords:
            # concat coord
            x_range = torch.linspace(-1, 1, cls_x.shape[-1], device=cls_x.device)
            y_range = torch.linspace(-1, 1, cls_x.shape[-2], device=cls_x.device)
            y_cls, x_cls = torch.meshgrid(y_range, x_range)
            y_reg, x_reg = torch.meshgrid(y_range, x_range)

            y_cls = y_cls.expand([cls_x.shape[0], 1, -1, -1])
            x_cls = x_cls.expand([cls_x.shape[0], 1, -1, -1])
            coord_feat_cls = torch.cat([x_cls, y_cls], 1)
            y_reg = y_reg.expand([cls_x.shape[0], 1, -1, -1])
            x_reg = x_reg.expand([cls_x.shape[0], 1, -1, -1])
            coord_feat_reg = torch.cat([x_reg, y_reg], 1)

            cls_x = torch.cat([cls_x, coord_feat_cls], 1)
            reg_x = torch.cat([reg_x, coord_feat_reg], 1)

        cls_conv2 = F.conv2d(cls_x, params[1]).relu()
        reg_conv2 = F.conv2d(reg_x, params[3]).relu()
        return cls_conv2, reg_conv2

    def init_weights(self):
        normal_init(self.downsampler_cls, std=0.01)
        normal_init(self.downsampler_reg, std=0.01)
