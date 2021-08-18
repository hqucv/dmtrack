import _init_paths
import neuron.data as data
from trackers import *
from mmcv import Config


if __name__ == '__main__':
    cfg_file = 'configs/dmtrackGS_dla34_fpn.py'
    cfg = Config.fromfile(cfg_file)
    ckp_file = 'work_dirs/dmtrack_dla34_fpn/dmtrackGS.pth'
    transforms = data.BasicPairTransforms(scale=cfg.data.test['scale'], train=cfg.data.test['train'])
    tracker = DMTrack(
        cfg_file, ckp_file, transforms,
        name_suffix='dmtrack_dla34_fpn')
    evaluators = [
        data.EvaluatorLaSOT(frame_stride=10),
    ]
    for e in evaluators:
        e.run(tracker, visualize=False, return_all=False)
        e.report(tracker.name, return_all=False)
