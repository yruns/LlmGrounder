import json

from tqdm import tqdm

SCANNET_TRAIN_SCANS = set(open("../../datasets/scannet/meta_data/scannetv2_train.txt").read().splitlines())
SCANNET_VAL_SCANS = set(open("../../datasets/scannet/meta_data/scannetv2_val.txt").read().splitlines())


def split_grounded3d():
    grounded3d_data = json.load(open("../../data/groundedscenecaption_format.json"))

    grounded3d_data_train = []
    grounded3d_data_val = []

    for data in tqdm(grounded3d_data):
        scan_id = data['scene_id']

        if scan_id in SCANNET_TRAIN_SCANS:
            grounded3d_data_train.append(data)
        elif scan_id in SCANNET_VAL_SCANS:
            grounded3d_data_val.append(data)
        else:
            print("Scan ID not found: ", scan_id)

    print("Number of training samples: ", len(grounded3d_data_train))
    print("Number of validation samples: ", len(grounded3d_data_val))
    json.dump(grounded3d_data_train, open("../../data/groundedscenecaption_format_train.json", 'w'), indent=4)
    json.dump(grounded3d_data_val, open("../../data/groundedscenecaption_format_val.json", 'w'), indent=4)


if __name__ == '__main__':
    split_grounded3d()
