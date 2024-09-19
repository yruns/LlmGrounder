ptv3_cfg = dict(
    engine="ptv3",

    # model settings
    model=dict(
        in_channels=6,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),

    # # dataset settings
    # dataset_type="ScanNetDataset" if num_classes == 20 else "ScanNet200Dataset",
    # data_root="data/ptv3_processed",
    num_classes=20,

    data=dict(
        train=dict(
            split="train",
            data_root="data/ptv3_processed",
            loop=1,
            transform=[
                dict(type="CenterShift", apply_z=True),
                dict(
                    type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
                ),
                dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                dict(type="RandomScale", scale=[0.9, 1.1]),
                dict(type="RandomFlip", p=0.5),
                dict(type="RandomJitter", sigma=0.005, clip=0.02),
                dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                dict(type="ChromaticJitter", p=0.95, std=0.05),
                dict(
                    type="GridSample",
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="train",
                    return_grid_coord=True,
                ),
                dict(type="SphereCrop", point_max=102400, mode="random"),
                dict(type="CenterShift", apply_z=False),
                dict(type="NormalizeColor"),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "segment"),
                    feat_keys=("color", "normal"),
                ),
            ],
            test_mode=False,
        ),
        val=dict(
            split="val",
            data_root="data/ptv3_processed",
            transform=[
                dict(type="CenterShift", apply_z=True),
                dict(
                    type="GridSample",
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="train",
                    return_grid_coord=True,
                ),
                dict(type="CenterShift", apply_z=False),
                dict(type="NormalizeColor"),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "segment"),
                    feat_keys=("color", "normal"),
                ),
            ],
            test_mode=False,
        )
    )
)
