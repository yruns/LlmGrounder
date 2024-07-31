"""
File: llava_nr3d.py
Date: 2024/7/28
Author: yruns

Description: This file contains ...
"""
import argparse
import copy
import json
import time
from tqdm import tqdm

from base_grounder.grounder import LMMGrounder
from base_grounder.utils import comm
from base_grounder.utils import ground as ground_utils
from base_grounder.utils import render as render_utils

if __name__ == '__main__':
    # add an argument
    parser = argparse.ArgumentParser(description='visprog nr3d.')
    parser.add_argument('--data_path', type=str, default='data/nr3d_val.json', help='exp name')
    parser.add_argument('--exp_name', type=str, default='test', help='exp name')
    args = parser.parse_args()

    ground_utils.lazy_load_feats(feat_path='/data2/shyue/ysh/paper-code/lang-point/ZSVG3D/data/scannet/feats_3d.pkl')

    with open(args.data_path, 'r') as f:
        datas = json.load(f)

    llava_grounder = LMMGrounder("llava-1.5-7b-hf", render_quality="high")

    correct_25 = 0
    correct_50 = 0
    correct_easy = 0
    correct_dep = 0
    easy_total = 0
    dep_total = 0
    recall = 0

    ground_errors = 0

    result = []

    try:
        for sample in tqdm(datas):
            obj_ids, inst_locs, center, _ = ground_utils.load_pc(sample["scan_id"])
            index = obj_ids.index(sample['target_id'])
            target_box = inst_locs[index]

            if sample['easy']:
                easy_total += 1
            if sample['view_dep']:
                dep_total += 1

            try:
                pred_box = llava_grounder.ground(sample)
                continue
                index, pred_box = llava_grounder.ask_gpt(sample)
                iou = comm.calc_iou(pred_box, target_box)

                new_sample = copy.deepcopy(sample)
                new_sample['pred_box'] = pred_box.tolist()
                new_sample['pred_index'] = index
                result.append(new_sample)

                if iou >= 0.25:
                    correct_25 += 1
                    # if iou >= 0.5:
                    #     correct_50 += 1
                    if sample['easy']:
                        correct_easy += 1
                    if sample['view_dep']:
                        correct_dep += 1

            except Exception as e:
                print(f"{sample['uid']}: {e}")
                ground_errors += 1
    finally:
        # json.dump(result, open('data/nr3d_val_gpt4o.json', 'w'), indent=4)
        json.dump(result, open('../data/nr3d_val.json', 'w'), indent=4)

        logger = comm.create_logger(str(args.exp_name + str(time.time())))

        logger.info('Easy {} {} / {}'.format(correct_easy / easy_total, correct_easy, easy_total))
        logger.info('Hard {} {} / {}'.format((correct_25 - correct_easy) / (len(datas) - easy_total),
                                             correct_25 - correct_easy,
                                             len(datas) - easy_total))
        logger.info('View-Dep {} {} / {}'.format(correct_dep / dep_total, correct_dep, dep_total))
        logger.info('View-Indep {} {} / {}'.format((correct_25 - correct_dep) / (len(datas) - dep_total),
                                                   correct_25 - correct_dep,
                                                   len(datas) - dep_total))
        logger.info('Acc@25 {} {} / {}'.format(correct_25 / len(datas), correct_25, len(datas)))
        # print('Acc@50', correct_50 / len(programs), correct_50, '/', len(programs))
        logger.info('Recall {} {} / {}'.format(recall / len(datas), recall, len(datas)))

        logger.info('Program Errors: {}'.format(ground_errors))
        logger.info('Programs errors rate: {}'.format(ground_errors / len(datas)))
