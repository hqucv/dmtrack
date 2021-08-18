# model settings
model = dict(
    type='DMTrack',
    backbone=dict(
        type='DLA',
        base_name='dla34'
        ),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512],
        out_channels=256, # 256
        start_level=0,
        add_extra_convs=False,
        extra_convs_on_inputs=False,
        num_outs=3,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='RTCondFCOSHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        out_in_channels=32,
        strides=[8, 16, 32],
        regress_ranges=((-1, 64), (64, 128), (128, 1e8)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        center_sampling=True
        ),
    visdom_cfg=dict(
        use_visdom=False,
        server='127.0.0.1',
        port=6007
    ),
    mid_channel=32,
    head_channels=256,  # 256
    relative_coords=False,
    featmap_num=3,
    temple_fusion=False
)

# model training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0,#0.05, # 0?
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)

# dataset settings
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='PairWrapper',
        ann_file=None,  # for compatibility
        # base_dataset='trackingnet_train, coco_train, got10k_train, lasot_train',
        base_dataset='coco_train, lasot_train',
        base_transforms='extra_full',
        # sampling_prob=[0.25, 0.25, 0.25, 0.25],
        sampling_prob=[0.5, 0.5],
        max_size=30000, # origin 30000
        max_instances=8, # 8
        with_label=True,
        sample_augment=True,
        augment_type='min',
        augment_start=256,
        augment_end=608,
        augment_stride=32,
        scale=(900, 608),
        #########
        # if use 'extra_partial/full_boost'
        gray_probability=0.05,
        sigma=[(2, 0.2), (0.2, 2), (3, 1), (1, 3), (2, 2)],
        blur_probability=0.05,
        #########
    ),
    test=dict(
        scale=(736, 512),
        train=False)
    )

# optimizer
optimizer = dict(type='SGD',
                 lr=0.001, #0.01
                 momentum=0.9,
                 weight_decay=0.0001,
                 paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
# runner configs
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=0.001/3,
    step=[44, 46])

total_epochs = 48

# default runtime
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]

# runtime settings
cudnn_benchmark = True
work_dir = 'work_dirs/dmtrack_dla34_fpn'
