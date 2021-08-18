import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)

import _init_paths
import argparse
import os
import torch

import mmdet
import neuron.ops as ops
from mmcv import Config
from mmdet.apis import get_root_logger, set_random_seed, \
    train_detector
from mmcv.runner import init_dist
from mmdet.models import build_detector
from mmdet.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a conditional convolution detector')
    parser.add_argument(
        '--config')
    parser.add_argument(
        '--work_dir')
    parser.add_argument('--load_from')
    parser.add_argument('--resume_from')
    parser.add_argument(
        '--base_dataset',  # names of training datasets, splitted by comma, see `datasets/wrappers` for options
        type=str)
    parser.add_argument(
        '--base_transforms',  # names of transforms, see `datasets/wrappers` for options
        type=str)
    parser.add_argument(
        '--sampling_prob',  # probabilities for sampling training datasets, splitted by comma, sum should be 1
        type=str)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--autoscale_lr', type=int, default=1)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--gpu_ids', default=[0,1])

    args = parser.parse_args()
    if not 'LOCAL_RANK' in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    # parse arguments
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    #if args.load_from is not None:
    cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.base_dataset is not None:
        cfg.data.train.base_dataset = args.base_dataset
    if args.base_transforms is not None:
        cfg.data.train.base_transforms = args.base_transforms
    if args.sampling_prob is not None:
        probs = [float(p) for p in args.sampling_prob.split(',')]
        cfg.data.train.sampling_prob = probs
    if args.fp16:
        cfg.fp16 = {'loss_scale': 512.}
    if args.workers is not None:
        cfg.data.workers_per_gpu = args.workers
    cfg.gpus = args.gpus
    if args.autoscale_lr:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8.
    if args.seed:
        cfg.seed = args.seed
    if args.gpu_ids:
        cfg.gpu_ids = args.gpu_ids
    ops.sys_print('Args:\n--', args)
    ops.sys_print('Configs:\n--', cfg)

    # init distributed env, logger and random seeds
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if args.seed is not None:
        set_random_seed(args.seed)

    # build model
    model = build_detector(
        cfg.model,
        train_cfg=cfg.train_cfg,
        test_cfg=cfg.test_cfg)

    # build dataset
    train_dataset = build_dataset(cfg.data.train)
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = {
            'mmdet_version': mmdet.__version__,
            'config': cfg.text,
            'CLASSES': train_dataset.CLASSES}
    model.CLASSES = train_dataset.CLASSES

    # run training
    train_detector(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        validate=args.validate)


if __name__ == '__main__':
    main()
