import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.roi_heads.roi_extractors import SingleRoIExtractor
from mmdet.core import bbox2roi
from mmcv.cnn import normal_init


__all__ = ['Template']


class FastNormalizedFusion(nn.Module):
    def __init__(self, in_nodes):
        super().__init__()
        self.in_nodes = in_nodes
        self.weight = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.register_buffer("eps", torch.tensor(0.0001))

    def forward(self, *x):
        if len(x) != self.in_nodes:
            raise RuntimeError(
                "Expected to have {} input nodes, but have {}.".format(self.in_nodes, len(x))
            )

        # where wi ≥ 0 is ensured by applying a relu after each wi (paper)
        weight = F.relu(self.weight)
        weighted_xs = [xi * wi for xi, wi in zip(x, weight)]
        normalized_weighted_x = sum(weighted_xs) / (weight.sum() + self.eps)
        return normalized_weighted_x


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


class Template(nn.Module):
    def __init__(self,
                 roi_out_size=7,
                 roi_sample_num=2,
                 channels=256,
                 strides=[4, 8, 16, 32],
                 featmap_num=5,
                 temple_fusion=False):
        super(Template, self).__init__()
        self.strides = strides
        self.temple_fusion = temple_fusion
        self.featmap_num = featmap_num
        if temple_fusion:
            self.roi_extractor = SingleRoIExtractor(
                roi_layer={
                    'type': 'RoIAlign',
                    'out_size': roi_out_size,
                    'sample_num': roi_sample_num},
                out_channels=channels,
                featmap_strides=[strides[0]])  # upsample to the first layer feature
            self.fpn_modulator = nn.ModuleList([
                nn.Conv2d(channels, channels, 1, padding=0)])
            self.weight = nn.Parameter(torch.ones(3, dtype=torch.float32))
            self.register_buffer("eps", torch.tensor(0.0001))
        else:
            self.roi_extractor = SingleRoIExtractor(
                roi_layer={
                    'type': 'RoIAlign',
                    'out_size': roi_out_size,
                    'sample_num': roi_sample_num},
                out_channels=channels,
                featmap_strides=strides)
            self.fpn_modulator = nn.ModuleList([
                nn.Conv2d(channels, channels, 1, padding=0)
                for _ in range(featmap_num)])

    def upsample_feat(self, feats_z):
        feats_z_upsample = []
        feats_z_upsample.append(feats_z[0])
        for feat, stride in zip(feats_z[1:], self.strides[1:]):
            scale_factor = stride / self.strides[0]
            feat_z_upsample = F.interpolate(feat, scale_factor=scale_factor, mode='aligned_bilinear')
            feats_z_upsample.append(feat_z_upsample)
        return feats_z_upsample

    def fuse_feat(self, feats_z, weight):
        feats_weight = []
        for index, feat in enumerate(feats_z):
            feat = weight[index] * feat
            feats_weight.append(feat)
        feat_fuse = feats_weight[0]
        if len(feats_weight) > 1:
            for fuse in feats_weight[1:]:
                feat_fuse += fuse
        return feat_fuse

    def forward(self, feats_z, gt_bboxes_z):
        rois = bbox2roi(gt_bboxes_z)
        if self.temple_fusion:
            feats_z = self.upsample_feat(feats_z)
            weight = F.relu(self.weight)
            feats_z = self.fuse_feat(feats_z, weight).unsqueeze(0)
            bbox_feats = self.roi_extractor(
                tuple(feats_z), rois)
        else:
            bbox_feats = self.roi_extractor(
                feats_z[:self.roi_extractor.num_inputs], rois)

        modulator = [bbox_feats[rois[:, 0] == j]
                     for j in range(len(gt_bboxes_z))]

        n_imgs = len(feats_z[0])
        for i in range(n_imgs):
            n_instances = len(modulator[i])
            for j in range(n_instances):
                query = modulator[i][j:j + 1]
                if self.temple_fusion:
                    out_ij = [self.fpn_modulator[0](query)]
                else:
                    out_ij = [self.fpn_modulator[k](query)
                              for k in range(self.featmap_num)]
                yield out_ij, i, j
        return

    def init_weights(self):
        for m in self.fpn_modulator:
            normal_init(m, std=0.01)


