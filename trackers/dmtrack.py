import torch
import numpy as np

from neuron.models import Tracker
import neuron.ops as ops
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.core import wrap_fp16_model

__all__ = ['DMTrack']


class DMTrack(Tracker):

    def __init__(self, cfg_file, ckp_file, transforms, name_suffix=''):
        name = 'DMTrack'
        if name_suffix:
            name += '_' + name_suffix
        super(DMTrack, self).__init__(
            name=name, is_deterministic=True)
        self.transforms = transforms

        # build config
        cfg = Config.fromfile(cfg_file)
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        self.cfg = cfg

        # build model
        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(
            model, ckp_file, map_location='cpu')
        model.CLASSES = ('object',)

        # GPU usage
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.model = model.to(self.device)

    @torch.no_grad()
    def init(self, img, bbox):
        self.model.eval()

        # prepare query data
        img_meta = {'ori_shape': img.shape}
        bboxes = np.expand_dims(bbox, axis=0)
        img, img_meta, bboxes = \
            self.transforms._process_query(img, img_meta, bboxes)
        img = img.unsqueeze(0).contiguous().to(
            self.device, non_blocking=True)
        bboxes = bboxes.to(self.device, non_blocking=True)

        # initialize the modulator
        self.model._process_query(img, [bboxes])

    @torch.no_grad()
    def update(self, img, return_all, **kwargs):
        self.model.eval()
        ori_img = img

        # prepare gallary data
        img_meta = {'ori_shape': img.shape}
        img, img_meta, _ = \
            self.transforms._process_gallary(img, img_meta, None)
        img = img.unsqueeze(0).contiguous().to(
            self.device, non_blocking=True)

        # get detections
        results = self.model._process_gallary(
            img, [img_meta], rescale=True, **kwargs)

        #ops.show_image_with_label(ori_img, results[0:10, :])

        if not return_all:
            # return the top-1 detection
            max_ind = results[:, -1].argmax()
            return results[max_ind, :4]
        else:
            if results.shape[0] < 100:
                fill_result = results[-1, :]
                fill_num = 100 - results.shape[0]
                for i in range(fill_num):
                    results = np.concatenate((results, fill_result[np.newaxis, :]))
            # return all detections
            return results[:, :4]