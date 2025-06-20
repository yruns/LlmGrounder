mask3d_cfg = {
    'engine': "mask3d",
    'data': {'train_mode': 'train', 'validation_mode': 'validation', 'test_mode': 'validation', 'ignore_label': 255,
             'add_raw_coordinates': True, 'add_colors': True, 'add_normals': False, 'in_channels': 3, 'num_labels': 200,
             'add_instance': True, 'task': 'instance_segmentation', 'pin_memory': False, 'num_workers': 4,
             'batch_size': 5, 'test_batch_size': 1, 'cache_data': False, 'voxel_size': 0.02, 'reps_per_epoch': 1,
             'cropping': False, 'cropping_args': {'min_points': 30000, 'aspect': 0.8, 'min_crop': 0.5, 'max_crop': 1.0},
             'crop_min_size': 20000, 'crop_length': 6.0, 'cropping_v1': True,
             'train_dataloader': {'shuffle': True, 'pin_memory': False, 'num_workers': 4, 'batch_size': 5},
             'validation_dataloader': {'shuffle': False, 'pin_memory': False, 'num_workers': 4, 'batch_size': 1},
             'test_dataloader': {'shuffle': False, 'pin_memory': False, 'num_workers': 4, 'batch_size': 1},
             'train_dataset': {'dataset_name': 'scannet200', 'data_dir': 'data/scannet_processed',
                               'image_augmentations_path': 'data/scannet_processed/albumentations_aug.json',
                               'volume_augmentations_path': 'data/scannet_processed/volumentations_aug.json',
                               'label_db_filepath': 'data/scannet_processed/label_database.json',
                               'color_mean_std': 'data/scannet_processed/color_mean_std.json', 'data_percent': 1.0,
                               'mode': 'train', 'ignore_label': 255, 'num_labels': 200, 'add_raw_coordinates': True,
                               'add_colors': True, 'add_normals': False, 'add_instance': True,
                               'instance_oversampling': 0.0, 'place_around_existing': False, 'point_per_cut': 0,
                               'max_cut_region': 0, 'flip_in_center': False, 'noise_rate': 0, 'resample_points': 0,
                               'add_unlabeled_pc': False, 'cropping': False,
                               'cropping_args': {'min_points': 30000, 'aspect': 0.8, 'min_crop': 0.5, 'max_crop': 1.0},
                               'is_tta': False, 'crop_min_size': 20000, 'crop_length': 6.0,
                               'filter_out_classes': [0, 2], 'label_offset': 2},
             'val_dataset': {'dataset_name': 'scannet200', 'data_dir': 'data/scannet_processed',
                             'image_augmentations_path': None, 'volume_augmentations_path': None,
                             'label_db_filepath': 'data/scannet_processed/label_database.json',
                             'color_mean_std': 'data/scannet_processed/color_mean_std.json',
                             'data_percent': 1.0, 'mode': 'validation', 'ignore_label': 255, 'num_labels': 200,
                             'add_raw_coordinates': True, 'add_colors': True, 'add_normals': False,
                             'add_instance': True, 'cropping': False, 'is_tta': False,
                             'crop_min_size': 20000, 'crop_length': 6.0, 'filter_out_classes': [0, 2],
                             'label_offset': 2},
             'test_dataset': {'dataset_name': 'scannet200', 'data_dir': 'data/scannet_processed',
                              'image_augmentations_path': None, 'volume_augmentations_path': None,
                              'label_db_filepath': 'data/scannet_processed/label_database.json',
                              'color_mean_std': 'data/scannet_processed/color_mean_std.json', 'data_percent': 1.0,
                              'mode': 'validation', 'ignore_label': 255, 'num_labels': 200, 'add_raw_coordinates': True,
                              'add_colors': True, 'add_normals': False, 'add_instance': True,
                              'cropping': False, 'is_tta': False, 'crop_min_size': 20000, 'crop_length': 6.0,
                              'filter_out_classes': [0, 2], 'label_offset': 2},
             'train_collation': {'ignore_label': 255, 'voxel_size': 0.02, 'mode': 'train', 'small_crops': False,
                                 'very_small_crops': False, 'batch_instance': False, 'probing': False,
                                 'task': 'instance_segmentation', 'ignore_class_threshold': 100,
                                 'filter_out_classes': [0, 2], 'label_offset': 2, 'num_queries': 100},
             'val_collation': {'ignore_label': 255, 'voxel_size': 0.02, 'mode': 'validation',
                               'batch_instance': False, 'probing': False, 'task': 'instance_segmentation',
                               'ignore_class_threshold': 100, 'filter_out_classes': [0, 2], 'label_offset': 2,
                               'num_queries': 100},
             'test_collation': {'ignore_label': 255, 'voxel_size': 0.02, 'mode': 'validation', 'batch_instance': False,
                                'probing': False, 'task': 'instance_segmentation', 'ignore_class_threshold': 100,
                                'filter_out_classes': [0, 2], 'label_offset': 2, 'num_queries': 100}}, 'logging': [
        {'_target_': 'pytorch_lightning.loggers.TensorBoardLogger', 'name': '${general.experiment_name}',
         'save_dir': '${general.save_dir}'}],
    'model': {'hidden_dim': 128, 'dim_feedforward': 1024, 'num_queries': 100, 'num_heads': 8, 'num_decoders': 3,
              'dropout': 0.0, 'pre_norm': False, 'use_level_embed': False, 'normalize_pos_enc': True,
              'positional_encoding_type': 'fourier', 'gauss_scale': 1.0, 'hlevels': [0, 1, 2, 3],
              'non_parametric_queries': True, 'random_query_both': False, 'random_normal': False,
              'random_queries': False, 'use_np_features': False, 'sample_sizes': [200, 800, 3200, 12800, 51200],
              'max_sample_size': False, 'shared_decoder': True, 'num_classes': 201, 'train_on_segments': True,
              'scatter_type': 'mean', 'voxel_size': 0.02, 'config': {
            'backbone': {'config': {'dialations': [1, 1, 1, 1], 'conv1_kernel_size': 5, 'bn_momentum': 0.02},
                         'in_channels': 3, 'out_channels': 200, 'out_fpn': True}}},
    'metrics': {'num_classes': 200, 'ignore_label': 255}, 'optimizer': {'lr': 0.0001},
    'scheduler': {'scheduler': {'max_lr': 0.0001, 'epochs': 601, 'steps_per_epoch': -1},
                  'pytorch_lightning_params': {'interval': 'step'}},
    'trainer': {'deterministic': False, 'max_epochs': 601, 'min_epochs': 1,
                'resume_from_checkpoint': 'saved/scannet200_val/last-epoch.ckpt', 'check_val_every_n_epoch': 50,
                'num_sanity_val_steps': 2}, 'callbacks': [
        {'_target_': 'pytorch_lightning.callbacks.ModelCheckpoint', 'monitor': 'val_mean_ap_50', 'save_last': True,
         'save_top_k': 1, 'mode': 'max', 'dirpath': '${general.save_dir}', 'filename': '{epoch}-{val_mean_ap_50:.3f}',
         'every_n_epochs': 1}, {'_target_': 'pytorch_lightning.callbacks.LearningRateMonitor'}],
    'matcher': {'cost_class': 2.0, 'cost_mask': 5.0, 'cost_dice': 2.0, 'num_points': -1},
    'loss': {'num_classes': 201, 'eos_coef': 0.1, 'losses': ['labels', 'masks'], 'num_points': -1,
             'oversample_ratio': 3.0, 'importance_sample_ratio': 0.75, 'class_weights': -1},
    'general': {'train_mode': True, 'task': 'instance_segmentation', 'seed': 42, 'checkpoint': None,
                'backbone_checkpoint': None, 'freeze_backbone': False, 'linear_probing_backbone': False,
                'train_on_segments': True, 'eval_on_segments': True, 'filter_out_instances': False,
                'save_visualizations': False, 'visualization_point_size': 20, 'decoder_id': -1, 'export': False,
                'use_dbscan': False, 'ignore_class_threshold': 100, 'project_name': 'scannet200',
                'workspace': 'jonasschult', 'experiment_name': 'scannet200_val', 'num_targets': 201,
                'add_instance': True, 'dbscan_eps': 0.95, 'dbscan_min_points': 1, 'export_threshold': 0.0001,
                'reps_per_epoch': 1, 'on_crops': False, 'scores_threshold': 0.0, 'iou_threshold': 1.0, 'area': 5,
                'eval_inner_core': -1, 'topk_per_image': 100, 'ignore_mask_idx': [], 'max_batch_size': 99999999,
                'save_dir': 'saved/${general.experiment_name}', 'gpus': 1}}
