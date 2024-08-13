"""
File: prepare_referit3d.py
Date: 2024/8/6
Author: yruns
"""
import json
import os.path as osp
import pandas as pd
import argparse
import glob
from tqdm import tqdm

from utils.referit3d import ReferIt3DUtils


def process_referit3d(referit3d_file, save_path, separate_storage=True, mentioned_only=True):
    basename = osp.basename(referit3d_file).split(".")[0]

    print("Now processing {}...".format(basename))
    processed_data = []
    # load data
    data = pd.read_csv(referit3d_file)
    if mentioned_only:
        data = data[data["mentions_target_class"]]
        print("{} samples are mentioned only, ignore not mentioned samples.".format(len(data)))

    train_split, val_split = ReferIt3DUtils.scannet_official_train_val("data/scannet/meta_data")

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        utterance = row["utterance"]
        if utterance.startswith("'") and utterance.endswith("'"):
            utterance = utterance[1:-1]

        scan_id, target_class, n_objects, target_id, distractors_ids = \
            ReferIt3DUtils.decode_stimulus_string(row["stimulus_id"])
        assert scan_id == row["scan_id"]
        assert target_class == row["instance_type"]
        assert target_id == row["target_id"]

        is_train = scan_id in train_split
        view_dep = ReferIt3DUtils.is_explicitly_view_dependent(row["tokens"])
        easy = len(distractors_ids) <= 2

        processed_data.append({
            "uid": idx,
            "scan_id": scan_id,
            "utterance": utterance,
            "target_class": target_class,
            "target_id": target_id,
            "distractors_ids": distractors_ids,
            "view_dep": view_dep,
            "easy": easy,
            "is_train": is_train,
        })

    with open(osp.join(save_path, "{}.json".format(basename)), "w") as f:
        json.dump(processed_data, f, indent=4)

    if separate_storage:
        train_samples = [entry for entry in processed_data if entry["is_train"]]
        with open(osp.join(save_path, "{}_train.json".format(basename)), "w") as f:
            json.dump(train_samples, f, indent=4)

        val_samples = [entry for entry in processed_data if not entry["is_train"]]
        with open(osp.join(save_path, "{}_val.json".format(basename)), "w") as f:
            json.dump(val_samples, f, indent=4)

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare Referit3D dataset'
    )

    parser.add_argument(
        "--root", default="data/referit3d",
        type=str, help="root path of Referit3D"
    )
    parser.add_argument(
        "--save_path", default="data/referit3d",
        type=str, help="save path of Referit3D"
    )

    args = parser.parse_args()

    referit3d_files = glob.glob(osp.join(args.root, r"*r3d.csv"))   # sr3d or nr3d
    for referit3d_file in referit3d_files:
        process_referit3d(referit3d_file, args.save_path)


