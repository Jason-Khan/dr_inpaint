norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='SwinInpaintor',
    disc_input_with_mask=True,
    encdec=dict(
        type='SwinuperEncoderDecoder',
        pretrained=None,
        backbone=dict(
            type='SwinTransformer',
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            ape=False,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            use_checkpoint=False,
            # pretrained='/home/wangk/inpaint/mmediting/models/swin_tiny_patch4_window7_224.pth'
        ),
        decode_head=dict(
            type='UPerHead',
            in_channels=[96, 192, 384, 768],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=192,
            dropout_ratio=0.1,
            num_classes=3,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(),
        ),
    ),
    disc=dict(
        type='MultiLayerDiscriminator',
        in_channels=4,
        max_channels=192,
        fc_in_channels=None,
        num_convs=6,
        norm_cfg=None,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
        out_act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
        with_spectral_norm=True,
    ),
    loss_gan=dict(
        type='GANLoss',
        gan_type='hinge',
        loss_weight=0.1,
    ),
    loss_l1_hole=dict(
        type='L1Loss',
        loss_weight=1.0,
    ),
    loss_l1_valid=dict(
        type='L1Loss',
        loss_weight=1.0,
    ),
    pretrained=None)

train_cfg = dict(disc_step=1)
test_cfg = dict(metrics=['l1', 'psnr', 'ssim'])

dataset_type = 'ImgInpaintingDataset'
input_shape = (256, 256)

train_pipeline = [
    dict(type='LoadImageFromFile', key='gt_img'),
    dict(
        type='LoadMask',
        mask_mode='irregular',
        mask_config=dict(
            num_vertexes=(4, 10),
            max_angle=6.0,
            length_range=(20, 128),
            brush_width=(10, 45),
            area_ratio_range=(0.15, 0.65),
            img_shape=input_shape)),
    dict(
        type='Crop',
        keys=['gt_img'],
        crop_size=(384, 384),
        random_crop=True,
    ),
    dict(
        type='Resize',
        keys=['gt_img'],
        scale=input_shape,
        keep_ratio=False,
    ),
    dict(
        type='Normalize',
        keys=['gt_img'],
        mean=[127.5] * 3,
        std=[127.5] * 3,
        to_rgb=False),
    dict(type='GetMaskedImage'),
    dict(
        type='Collect',
        keys=['gt_img', 'masked_img', 'mask'],
        meta_keys=['gt_img_path']),
    dict(type='ImageToTensor', keys=['gt_img', 'masked_img', 'mask'])
]

test_pipeline = train_pipeline

data_root = 'data/places365_standard'

data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=2, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=f'{data_root}/train.txt',
        data_prefix=data_root,
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(
        type=dataset_type,
        ann_file=f'{data_root}/val.txt',
        data_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file=(f'{data_root}/val.txt'),
        data_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True))

optimizers = dict(
    generator=dict(type='Adam', lr=0.0001), disc=dict(type='Adam', lr=0.0001))

lr_config = dict(policy='Fixed', by_epoch=False)

checkpoint_config = dict(by_epoch=False, interval=5000)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit'))
    ])

visual_config = dict(
    type='VisualizationHook',
    output_dir='visual',
    interval=1000,
    res_name_list=[
        'gt_img', 'masked_img', 'fake_res', 'fake_img'
    ],
)

evaluation = dict(interval=50000)

total_iters = 500000
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/train_swinuper'
load_from = None
resume_from = work_dir+'/latest.pth'
# resume_from = None
workflow = [('train', 10000)]
exp_name = 'swinuper'
find_unused_parameters = False
