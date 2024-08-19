"""
File: referit3d.py
Date: 2024/8/6
Author: yruns
"""
import os.path as osp


class ReferIt3DUtils:

    @staticmethod
    def scannet_official_train_val(scannet_meta, valid_views=None, verbose=True):
        """
        :param scannet_meta:
        :param verbose:
        :param valid_views: None or list like ["00", "01"]
        :return:
        """
        train_split = osp.join(scannet_meta, "scannetv2_train.txt")
        train_split = open(train_split).read().splitlines()
        val_split = osp.join(scannet_meta, "scannetv2_val.txt")
        val_split = open(val_split).read().splitlines()

        if valid_views is not None:
            train_split = [sc for sc in train_split if sc[-2:] in valid_views]
            val_split = [sc for sc in val_split if sc[-2:] in valid_views]

        if verbose:
            print('#train/test scans:', len(train_split), '/', len(val_split))

        return train_split, val_split

    @staticmethod
    def is_explicitly_view_dependent(tokens):
        if isinstance(tokens, str):
            tokens = eval(tokens)
        target_words = {
            'front', 'behind', 'back', 'right', 'left', 'facing', 'leftmost', 'rightmost',
            'looking', 'across'
        }
        return len(set(tokens).intersection(target_words)) > 0

    @staticmethod
    def decode_stimulus_string(s):
        """
        Split into scene_id, instance_label, # objects, target object id,
        distractors object id.

        :param s: the stimulus string
        """
        if len(s.split('-', maxsplit=4)) == 4:
            scene_id, instance_label, n_objects, target_id = \
                s.split('-', maxsplit=4)
            distractors_ids = ""
        else:
            scene_id, instance_label, n_objects, target_id, distractors_ids = \
                s.split('-', maxsplit=4)

        instance_label = instance_label.replace('_', ' ')
        n_objects = int(n_objects)
        target_id = int(target_id)
        distractors_ids = [int(i) for i in distractors_ids.split('-') if i != '']
        assert len(distractors_ids) == n_objects - 1

        return scene_id, instance_label, n_objects, target_id, distractors_ids
