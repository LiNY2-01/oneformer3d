_base_ = [
    'mmdet3d::_base_/default_runtime.py',
    # 'mmdet3d::_base_/datasets/scannet-seg.py'
]
custom_imports = dict(imports=['oneformer3d'])

# model settings
num_channels = 64
num_instance_classes = 21
num_semantic_classes = 21

model = dict(
    type="S3DISOneFormer3D",
    data_preprocessor=dict(type="Det3DDataPreprocessor_"),
    in_channels=6,
    num_channels=num_channels,
    voxel_size=0.05,
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    # query_thr=0.5,
    backbone=dict(
        type="SpConvUNet",
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True,
    ),
    decoder=dict(
        type="QueryDecoder",
        num_layers=3,
        num_classes=num_instance_classes,
        num_instance_queries=250,
        num_semantic_queries=num_semantic_classes,
        num_instance_classes=num_instance_classes,
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn="gelu",
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=True,
    ),
    criterion=dict(
        type='S3DISUnifiedCriterion',
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(
            type='S3DISSemanticCriterion',
            loss_weight=5.0),
        inst_criterion=dict(
            type='InstanceCriterion',
            matcher=dict(
                type='HungarianMatcher',
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0)]),
            loss_weight=[0.5, 1.0, 1.0, 0.5],
            num_classes=num_instance_classes,
            non_object_weight=0.05,
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=90,
        inst_score_thr=0.1,
        pan_score_thr=0.4,
        npoint_thr=300,
        obj_normalization=True,
        obj_normalization_thr=0.01,
        sp_score_thr=0.15,
        nms=True,
        matrix_nms_kernel='linear',
        num_sem_cls=num_semantic_classes,
        stuff_cls=[1,2],
        thing_cls=[3,4,5,6,7,8,9, 10, 11]),
)

class_names = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "ceiling",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "other",
]
metainfo = dict(classes=class_names)
# dataset settings
dataset_type = "Matterport3DDataset"
data_root = "data/mp3d_0223/"
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask',)
input_modality = dict(use_lidar=True, use_camera=False)
# sp_pts_mask='super_points')

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
    ),
    dict(
        type="LoadAnnotations3D_",
        with_bbox_label_3d=True,
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=False,
    ),
    # dict(type="PointSample_", num_points=200000),
    dict(type="PointSegClassMapping"),
    dict(type="PointInstClassMappingMp3d_"),
    dict(
        type="RandomFlip3D",
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
    ),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False,
    ),
    dict(
        type="NormalizePointsColor_",
        color_mean=[127.5, 127.5, 127.5],
        color_std=None,
        no_color=False,
    ),
    # dict(
    #     type='AddSuperPointAnnotations',
    #     num_classes=num_semantic_classes,
    #     stuff_classes=[0, 1],
    #     merge_non_stuff_cls=False),
    # dict(
    #     type='ElasticTransfrom',
    #     gran=[6, 20],
    #     mag=[40, 160],
    #     voxel_size=0.02,
    #     p=0.5),
    dict(
        type="Pack3DDetInputs_",
        keys=[
            "points",
            "gt_labels_3d",
            "pts_semantic_mask",
            "pts_instance_mask",
            "gt_sp_masks",
            "elastic_coords",
        ],
    ),
]
test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
    ),
    dict(
        type="LoadAnnotations3D_",
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=False,
    ),
    dict(type="PointSegClassMapping"),
    dict(
        type="PointInstClassMappingMp3d_",
    ),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="NormalizePointsColor_",
                color_mean=[127.5, 127.5, 127.5],
                color_std=None,
                no_color=False,
            ),
            # dict(
            #     type='AddSuperPointAnnotations',
            #     num_classes=num_semantic_classes,
            #     stuff_classes=[0, 1],
            #     merge_non_stuff_cls=False),
        ],
    ),
    dict(
        type="Pack3DDetInputs_",
        keys=[
            "points",
        ],
    ),
]

eval_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
    ),
    dict(
        type="LoadAnnotations3D_",
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=False,
    ),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="NormalizePointsColor_",
                color_mean=[127.5, 127.5, 127.5],
                color_std=None,
                no_color=False,
            ),
        ],
    ),
    dict(
        type="Pack3DDetInputs_",
        keys=[
            "points",
        ],
    ),
]

# run settings
train_dataloader = dict(
    batch_size=2,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file="matterport3d_oneformer3d_infos_train.pkl",
        data_root=data_root,
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        ignore_index=num_semantic_classes,
        metainfo=metainfo,
        scene_idxs=None,
        test_mode=False,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file="matterport3d_oneformer3d_infos_val.pkl",
        data_root=data_root,
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        metainfo=metainfo,
        ignore_index=num_semantic_classes,
        test_mode=True,
    ),
)

eval_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file="matterport3d_oneformer3d_infos_val.pkl",
        data_root=data_root,
        data_prefix=data_prefix,
        pipeline=eval_pipeline,
        metainfo=metainfo,
        ignore_index=num_semantic_classes,
        test_mode=True,
    ),
)
test_dataloader = val_dataloader
# eval_dataloader = test_dataloader

class_names_all = class_names +  ['unlabeled']

label2cat = {i: name for i, name in enumerate(class_names_all)}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[num_semantic_classes],
    classes=class_names_all,
    dataset_name="Matterport3D",
)

sem_mapping = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 22, 24, 28, 33, 34, 36, 39]
inst_mapping = sem_mapping[2:]
val_evaluator = dict(
    type='UnifiedSegMetric',
    stuff_class_inds=[], 
    thing_class_inds=list(range(num_semantic_classes)), 
    min_num_points=1, 
    id_offset=2**16,
    sem_mapping=sem_mapping,
    inst_mapping=inst_mapping,
    metric_meta=metric_meta)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.00005, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2),
    accumulative_counts=2,
)

param_scheduler = dict(type='PolyLR', begin=0, end=200, power=0.9)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
default_hooks = dict(
    checkpoint=dict(
        interval=16, max_keep_ckpts=1, save_best=["all_ap_50%", "miou"], rule="greater"
    ),
)

load_from = "work_dirs/tmp/instance-only-oneformer3d_1xb2_scannet-and-structured3d.pth"
# load_from = "work_dirs/oneformer3d_1xb4_mp3d/epoch_x_320.pth"
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="Det3dInstanceVisualizer", vis_backends=vis_backends, name="visualizer"
)
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=200,
    val_interval=1,
    dynamic_intervals=[(3, 16), (200 - 16, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
