import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

__all__ = ['Cond_PreModulator']


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


class Cond_PreModulator(nn.Module):

    def __init__(self, relative_coords=True):
        super(Cond_PreModulator, self).__init__()
        self.relative_coords = relative_coords

    def forward(self, cls_x, reg_x, params):
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

        cls_conv1 = F.conv2d(cls_x, params[0]).relu()
        reg_conv1 = F.conv2d(reg_x, params[2]).relu()
        return cls_conv1, reg_conv1
