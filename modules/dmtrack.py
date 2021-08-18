import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
import torch.nn as nn

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.core import auto_fp16, get_classes, tensor2imgs, \
    bbox2result, bbox2roi, build_assigner, build_sampler, multi_apply
from .cond_conv import Cond_Conv
from .template import Template
from .cond_modulator import Cond_Modulator
from .cond_premodulator import Cond_PreModulator


import time

from tools.visdom_server import Visdom

__all__ = ['DMTrack']


def visdom_ui_handler(self, data):
    if data['event_type'] == 'KeyPress':
        if data['key'] == ' ':
            self.pause_mode = not self.pause_mode

        elif data['key'] == 'ArrowRight' and self.pause_mode:
            self.step = True


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


@DETECTORS.register_module()
class DMTrack(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 visdom_cfg,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 mid_channel=256,
                 head_channels=256,
                 relative_coords=False,
                 featmap_num=3,
                 temple_fusion=False):
        super(DMTrack, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.mid_channel = mid_channel
        self.relative_coords = relative_coords
        # template roi feature
        self.template = Template(roi_out_size=7,
                                 roi_sample_num=2,
                                 channels=head_channels,
                                 strides=[8, 16, 32],# for resnet [4, 8, 16, 32],
                                 featmap_num=featmap_num,
                                 temple_fusion=temple_fusion)
        # build Controller
        self.controller = Cond_Conv(feat_channels=head_channels, modulator_channel=self.mid_channel, filter_size=1, relative_coords=self.relative_coords)
        # build Conditional Modulator
        self.cond_premodulator = Cond_PreModulator(relative_coords=self.relative_coords)
        # build Conditional Modulator
        self.cond_modulator = Cond_Modulator(feat_channel=head_channels, modulator_channel=self.mid_channel, relative_coords=self.relative_coords)
        # initialize weights
        self.template.init_weights()
        self.controller.init_weights()
        self.cond_modulator.init_weights()

        # visualize
        visdom_info = {
            'use_visdom': visdom_cfg.use_visdom,
            'server': visdom_cfg.server,
            'port': visdom_cfg.port
        }
        self.visdom_server = None
        if visdom_cfg.use_visdom:
            self.visdom_server = Visdom(True, {'handler': visdom_ui_handler, 'win_id': 'Training'},
                                        visdom_info=visdom_info)

    @auto_fp16(apply_to=('img_z', 'img_x'))
    def forward(self,
                img_z,
                img_x,
                img_meta_z,
                img_meta_x,
                return_loss=True,
                **kwargs):
        if return_loss:
            return self.forward_train(
                img_z, img_x, img_meta_z, img_meta_x, **kwargs)
        else:
            return self.forward_test(
                img_z, img_x, img_meta_z, img_meta_x, **kwargs)

    def forward_dummy(self, *args, **kwargs):
        raise NotImplementedError(
            'forward_dummy is not implemented for QG_RCNN')

    def transfrom_params(self, params, feat_channel=256, controller_channels=8, filter_size=1):
        cond_params = params
        param_list = []
        param_level0 = []
        param_level1 = []
        param_level2 = []
        for type_index, param in enumerate(cond_params):
            for level_index, level_param in enumerate(param):
                if type_index == 0:
                    level_param = level_param.view(feat_channel, feat_channel, filter_size, filter_size)
                elif type_index == 1:
                    level_param = level_param.view(controller_channels, controller_channels, filter_size, filter_size)
                elif type_index == 2:
                    level_param = level_param.view(feat_channel, feat_channel, filter_size, filter_size)
                elif type_index == 3:
                    level_param = level_param.view(controller_channels, controller_channels, filter_size, filter_size)

                if level_index == 0:
                    param_level0.append(level_param)
                elif level_index == 1:
                    param_level1.append(level_param)
                elif level_index == 2:
                    param_level2.append(level_param)

        param_list.append(param_level0)
        param_list.append(param_level1)
        param_list.append(param_level2)
        return param_list

    def pre_modula(self, x, param):
        cls_feat = x
        reg_feat = x

        cls_feat, reg_feat = self.cond_premodulator(cls_feat, reg_feat, param)
        return (cls_feat, reg_feat)


    def forward_train(self,
                      img_z,
                      img_x,
                      img_meta_z,
                      img_meta_x,
                      gt_bboxes_z,
                      gt_bboxes_x,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        # vis annotation
        '''
        from PIL import Image, ImageDraw
        import torchvision
        transform = UnNormalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))
        img_x = transform(img_x)
        img_z = transform(img_z)
        image_x = np.transpose(img_x[0].cpu().numpy(), (1,2,0))
        image_z = np.transpose(img_z[0].cpu().numpy(), (1,2,0))
        image_x = Image.fromarray(np.uint8(image_x))
        image_z = Image.fromarray(np.uint8(image_z))
        drawx = ImageDraw.Draw(image_x)
        for gt in gt_bboxes_x[0]:
            #drawx.rectangle([gt[0], gt[1], gt[0]+gt[2], gt[1]+gt[3]])
            drawx.rectangle([gt[0], gt[1], gt[2], gt[3]])
        drawz = ImageDraw.Draw(image_z)
        for gt in gt_bboxes_z[0]:
            #drawz.rectangle([gt[0], gt[1], gt[0]+gt[2], gt[1]+gt[3]])
            drawz.rectangle([gt[0], gt[1], gt[2], gt[3]])
        image_x.show()
        image_z.show()
        '''
        z = self.extract_feat(img_z)
        x = self.extract_feat(img_x)

        losses = {}
        total = 0.
        for z_ij, i, j in self.template(z, gt_bboxes_z):
            losses_ij = {}

            # feature
            x_ij = [u[i:i + 1] for u in x]
            # select the j-th bbox/meta/label of the i-th image
            gt_bboxes_ij = gt_bboxes_x[i:i + 1]
            gt_bboxes_ij[0] = gt_bboxes_ij[0][j:j + 1]
            gt_labels_ij = gt_labels[i:i + 1]
            gt_labels_ij[0] = gt_labels_ij[0][j:j + 1]
            img_meta_xi = img_meta_x[i:i + 1]

            cond_conv_params = multi_apply(self.controller, z_ij)
            cond_conv_params = self.transfrom_params(cond_conv_params, feat_channel=256, controller_channels=self.mid_channel, filter_size=1)
            #modula_x = multi_apply(self.cond_modulator, outs, cond_conv_params)
            #modula_x = [x.unsqueeze(0) for x in modula_x[0]]
            x_ij_cls, x_ij_reg = multi_apply(self.pre_modula, x_ij, cond_conv_params)
            outs = self.bbox_head(x_ij_cls, x_ij_reg, cond_conv_params, self.cond_modulator)

            '''
            from PIL import Image, ImageDraw
            import torchvision
            transform = UnNormalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))
            img_x_show = transform(img_x.clone())
            img_z_show = transform(img_z.clone())
            image_x_show = np.transpose(img_x_show[0].cpu().numpy(), (1,2,0))
            image_z_show = np.transpose(img_z_show[0].cpu().numpy(), (1,2,0))
            image_x_show = Image.fromarray(np.uint8(image_x_show))
            image_z_show = Image.fromarray(np.uint8(image_z_show))
            drawx = ImageDraw.Draw(image_x_show)
            for gt in gt_bboxes_ij[0]:
                drawx.rectangle([gt[0], gt[1], gt[2], gt[3]], outline='green', width=5)
            outs_show = self.bbox_head(x_ij, cond_conv_params, self.cond_modulator)
            bbox_list = self.bbox_head.get_bboxes(
                *outs_show, img_meta_xi
            )
            for box in [bbox_list[0][0][0]]:
                drawx.rectangle([box[0], box[1], box[2], box[3]], outline='red', width=5)

            image_x_show.show()
            time.sleep(8)
            '''

            loss_inputs = outs + (gt_bboxes_ij, gt_labels_ij, img_meta_xi, self.train_cfg)
            loss_bbox = self.bbox_head.loss(*loss_inputs)
            losses_ij.update(loss_bbox)

            # update losses
            for k, v in losses_ij.items():
                if k in losses:
                    if isinstance(v, (tuple, list)):
                        for u in range(len(v)):
                            losses[k][u] += v[u]
                    else:
                        losses[k] += v
                else:
                    losses[k] = v
            total += 1.

            if self.visdom_server is not None:
                transform = UnNormalize(mean=(0.408, 0.447, 0.470), std=(0.289, 0.274, 0.278))
                outs_show = self.bbox_head(x_ij, cond_conv_params, self.cond_modulator)
                bbox_list = self.bbox_head.get_bboxes(
                    *outs_show, img_meta_xi
                )
                self.visdom_server.register((transform,
                                             img_z[i],
                                             gt_bboxes_z[i][j],
                                             img_x[i],
                                             gt_bboxes_z[i][j],
                                             bbox_list[0][0]
                                             ),
                                     'Training', 1, 'Training')
                time.sleep(2)

        # average the losses over instances
        for k, v in losses.items():
            if isinstance(v, (tuple, list)):
                for u in range(len(v)):
                    losses[k][u] /= total
            else:
                losses[k] /= total

        return losses


    def aug_test(self, *args, **kwargs):
        raise NotImplementedError(
            'aug_test is not implemented for QG_RCNN')

    def show_result(self, *args, **kwargs):
        raise NotImplementedError(
            'show_result is not implemented for QG_RCNN')

    def _process_query(self, img_z, gt_bboxes_z):
        self._query = self.extract_feat(img_z)
        self._gt_bboxes_z = gt_bboxes_z
        # generate cond conv
        z_feats = next(self.template(self._query, self._gt_bboxes_z))[0]
        cond_conv_params = multi_apply(self.controller, z_feats)
        self.cond_conv_params = self.transfrom_params(cond_conv_params,
                                                      feat_channel=256,
                                                      controller_channels=self.mid_channel,
                                                      filter_size=1)

    def _process_gallary(self, img_x, img_meta_x, **kwargs):
        x = self.extract_feat(img_x)
        x_00 = [u[0:1] for u in x]
        x_ij_cls, x_ij_reg = multi_apply(self.pre_modula, x_00, self.cond_conv_params)

        outs = self.bbox_head(x_ij_cls, x_ij_reg, self.cond_conv_params, self.cond_modulator)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_meta_x, **kwargs
        )
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]

        return bbox_results[0][1]
