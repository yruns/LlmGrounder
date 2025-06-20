mask3d_cfg = {
    'data': {'train_mode': 'train', 'validation_mode': 'validation', 'test_mode': 'validation', 'ignore_label': 255,
             'add_raw_coordinates': True, 'add_colors': True, 'add_normals': False, 'in_channels': 3, 'num_labels': 200,
             'add_instance': '${general.add_instance}', 'task': '${general.task}', 'pin_memory': False,
             'num_workers': 4, 'batch_size': 5, 'test_batch_size': 1, 'cache_data': False, 'voxel_size': 0.02,
             'reps_per_epoch': '${general.reps_per_epoch}', 'cropping': False,
             'cropping_args': {'min_points': 30000, 'aspect': 0.8, 'min_crop': 0.5, 'max_crop': 1.0},
             'crop_min_size': 20000, 'crop_length': 6.0, 'cropping_v1': True,
             'train_dataloader': {'_target_': 'torch.utils.data.DataLoader', 'shuffle': True,
                                  'pin_memory': '${data.pin_memory}', 'num_workers': '${data.num_workers}',
                                  'batch_size': '${data.batch_size}'},
             'validation_dataloader': {'_target_': 'torch.utils.data.DataLoader', 'shuffle': False,
                                       'pin_memory': '${data.pin_memory}', 'num_workers': '${data.num_workers}',
                                       'batch_size': '${data.test_batch_size}'},
             'test_dataloader': {'_target_': 'torch.utils.data.DataLoader', 'shuffle': False,
                                 'pin_memory': '${data.pin_memory}', 'num_workers': '${data.num_workers}',
                                 'batch_size': '${data.test_batch_size}'},
             'train_dataset': {'_target_': 'datasets.semseg.SemanticSegmentationDataset', 'dataset_name': 'scannet200',
                               'data_dir': 'data/processed/scannet200',
                               'image_augmentations_path': 'conf/augmentation/albumentations_aug.yaml',
                               'volume_augmentations_path': 'conf/augmentation/volumentations_aug.yaml',
                               'label_db_filepath': 'data/processed/scannet200/label_database.yaml',
                               'color_mean_std': 'data/processed/scannet200/color_mean_std.yaml', 'data_percent': 1.0,
                               'mode': '${data.train_mode}', 'ignore_label': '${data.ignore_label}',
                               'num_labels': '${data.num_labels}', 'add_raw_coordinates': '${data.add_raw_coordinates}',
                               'add_colors': '${data.add_colors}', 'add_normals': '${data.add_normals}',
                               'add_instance': '${data.add_instance}', 'instance_oversampling': 0.0,
                               'place_around_existing': False, 'point_per_cut': 0, 'max_cut_region': 0,
                               'flip_in_center': False, 'noise_rate': 0, 'resample_points': 0,
                               'add_unlabeled_pc': False, 'cropping': '${data.cropping}',
                               'cropping_args': '${data.cropping_args}', 'is_tta': False,
                               'crop_min_size': '${data.crop_min_size}', 'crop_length': '${data.crop_length}',
                               'filter_out_classes': [0, 2], 'label_offset': 2},
             'validation_dataset': {'_target_': 'datasets.semseg.SemanticSegmentationDataset',
                                    'dataset_name': 'scannet200', 'data_dir': 'data/processed/scannet200',
                                    'image_augmentations_path': None, 'volume_augmentations_path': None,
                                    'label_db_filepath': 'data/processed/scannet200/label_database.yaml',
                                    'color_mean_std': 'data/processed/scannet200/color_mean_std.yaml',
                                    'data_percent': 1.0, 'mode': '${data.validation_mode}',
                                    'ignore_label': '${data.ignore_label}', 'num_labels': '${data.num_labels}',
                                    'add_raw_coordinates': '${data.add_raw_coordinates}',
                                    'add_colors': '${data.add_colors}', 'add_normals': '${data.add_normals}',
                                    'add_instance': '${data.add_instance}', 'cropping': False, 'is_tta': False,
                                    'crop_min_size': '${data.crop_min_size}', 'crop_length': '${data.crop_length}',
                                    'filter_out_classes': [0, 2], 'label_offset': 2},
             'test_dataset': {'_target_': 'datasets.semseg.SemanticSegmentationDataset', 'dataset_name': 'scannet200',
                              'data_dir': 'data/processed/scannet200', 'image_augmentations_path': None,
                              'volume_augmentations_path': None,
                              'label_db_filepath': 'data/processed/scannet200/label_database.yaml',
                              'color_mean_std': 'data/processed/scannet200/color_mean_std.yaml', 'data_percent': 1.0,
                              'mode': '${data.test_mode}', 'ignore_label': '${data.ignore_label}',
                              'num_labels': '${data.num_labels}', 'add_raw_coordinates': '${data.add_raw_coordinates}',
                              'add_colors': '${data.add_colors}', 'add_normals': '${data.add_normals}',
                              'add_instance': '${data.add_instance}', 'cropping': False, 'is_tta': False,
                              'crop_min_size': '${data.crop_min_size}', 'crop_length': '${data.crop_length}',
                              'filter_out_classes': [0, 2], 'label_offset': 2},
             'train_collation': {'_target_': 'datasets.utils.VoxelizeCollate', 'ignore_label': '${data.ignore_label}',
                                 'voxel_size': '${data.voxel_size}', 'mode': '${data.train_mode}', 'small_crops': False,
                                 'very_small_crops': False, 'batch_instance': False,
                                 'probing': '${general.linear_probing_backbone}', 'task': '${general.task}',
                                 'ignore_class_threshold': '${general.ignore_class_threshold}',
                                 'filter_out_classes': '${data.train_dataset.filter_out_classes}',
                                 'label_offset': '${data.train_dataset.label_offset}',
                                 'num_queries': '${model.num_queries}'},
             'validation_collation': {'_target_': 'datasets.utils.VoxelizeCollate',
                                      'ignore_label': '${data.ignore_label}', 'voxel_size': '${data.voxel_size}',
                                      'mode': '${data.validation_mode}', 'batch_instance': False,
                                      'probing': '${general.linear_probing_backbone}', 'task': '${general.task}',
                                      'ignore_class_threshold': '${general.ignore_class_threshold}',
                                      'filter_out_classes': '${data.validation_dataset.filter_out_classes}',
                                      'label_offset': '${data.validation_dataset.label_offset}',
                                      'num_queries': '${model.num_queries}'},
             'test_collation': {'_target_': 'datasets.utils.VoxelizeCollate', 'ignore_label': '${data.ignore_label}',
                                'voxel_size': '${data.voxel_size}', 'mode': '${data.test_mode}',
                                'batch_instance': False, 'probing': '${general.linear_probing_backbone}',
                                'task': '${general.task}',
                                'ignore_class_threshold': '${general.ignore_class_threshold}',
                                'filter_out_classes': '${data.test_dataset.filter_out_classes}',
                                'label_offset': '${data.test_dataset.label_offset}',
                                'num_queries': '${model.num_queries}'}}, 'logging': [
        {'_target_': 'pytorch_lightning.loggers.TensorBoardLogger', 'name': '${general.experiment_name}',
         'save_dir': '${general.save_dir}'}],
    'model': {'_target_': 'models.Mask3D', 'hidden_dim': 128, 'dim_feedforward': 1024, 'num_queries': 100,
              'num_heads': 8, 'num_decoders': 3, 'dropout': 0.0, 'pre_norm': False, 'use_level_embed': False,
              'normalize_pos_enc': True, 'positional_encoding_type': 'fourier', 'gauss_scale': 1.0,
              'hlevels': [0, 1, 2, 3], 'non_parametric_queries': True, 'random_query_both': False,
              'random_normal': False, 'random_queries': False, 'use_np_features': False,
              'sample_sizes': [200, 800, 3200, 12800, 51200], 'max_sample_size': False, 'shared_decoder': True,
              'num_classes': '${general.num_targets}', 'train_on_segments': '${general.train_on_segments}',
              'scatter_type': 'mean', 'voxel_size': '${data.voxel_size}', 'config': {
            'backbone': {'_target_': 'models.Res16UNet34C',
                         'config': {'dialations': [1, 1, 1, 1], 'conv1_kernel_size': 5, 'bn_momentum': 0.02},
                         'in_channels': '${data.in_channels}', 'out_channels': '${data.num_labels}', 'out_fpn': True}}},
    'metrics': {'_target_': 'models.metrics.ConfusionMatrix', 'num_classes': '${data.num_labels}',
                'ignore_label': '${data.ignore_label}'}, 'optimizer': {'_target_': 'torch.optim.AdamW', 'lr': 0.0001},
    'scheduler': {'scheduler': {'_target_': 'torch.optim.lr_scheduler.OneCycleLR', 'max_lr': '${optimizer.lr}',
                                'epochs': '${trainer.max_epochs}', 'steps_per_epoch': -1},
                  'pytorch_lightning_params': {'interval': 'step'}},
    'trainer': {'deterministic': False, 'max_epochs': 601, 'min_epochs': 1,
                'resume_from_checkpoint': 'saved/scannet200_val/last-epoch.ckpt', 'check_val_every_n_epoch': 50,
                'num_sanity_val_steps': 2}, 'callbacks': [
        {'_target_': 'pytorch_lightning.callbacks.ModelCheckpoint', 'monitor': 'val_mean_ap_50', 'save_last': True,
         'save_top_k': 1, 'mode': 'max', 'dirpath': '${general.save_dir}', 'filename': '{epoch}-{val_mean_ap_50:.3f}',
         'every_n_epochs': 1}, {'_target_': 'pytorch_lightning.callbacks.LearningRateMonitor'}],
    'matcher': {'_target_': 'models.matcher.HungarianMatcher', 'cost_class': 2.0, 'cost_mask': 5.0, 'cost_dice': 2.0,
                'num_points': -1},
    'loss': {'_target_': 'models.criterion.SetCriterion', 'num_classes': '${general.num_targets}', 'eos_coef': 0.1,
             'losses': ['labels', 'masks'], 'num_points': '${matcher.num_points}', 'oversample_ratio': 3.0,
             'importance_sample_ratio': 0.75, 'class_weights': -1},
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

from typing import Dict


def process(cfg: Dict, global_dict):
    new_dict = dict()

    for key, value in cfg.items():
        if isinstance(value, dict):
            new_dict[key] = process(value, global_dict)
        elif isinstance(value, str) and value.startswith('${'):
            value = value.replace('${', '').replace('}', '')
            if key == "num_classes":
                print("here")

            chain = value.split('.')

            cur_global_dict = global_dict
            while len(chain) > 0:
                key_ = chain.pop(0)
                value = cur_global_dict[key_]
                cur_global_dict = value
            new_dict[key] = value
        elif key == "_target_":
            continue
        else:
            new_dict[key] = value

    return new_dict


if __name__ == '__main__':
    new_dict = process(mask3d_cfg, mask3d_cfg)
    print(new_dict)
