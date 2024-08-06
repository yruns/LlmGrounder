"""
File: llava_nr3d.py
Date: 2024/7/28
Author: yruns


"""
import argparse
import copy
import json
import time
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

from grounder import LMMGrounder
from base_grounder.utils import comm, ground as ground_utils

if __name__ == '__main__':
    # add an argument
    parser = argparse.ArgumentParser(description='visprog nr3d.')
    parser.add_argument('--data_path', type=str, default='data/nr3d_val.json', help='exp name')
    parser.add_argument('--exp_name', type=str, default='test', help='exp name')
    args = parser.parse_args()

    ground_utils.lazy_load_feats(feat_path='/data2/shyue/ysh/paper-code/lang-point/ZSVG3D/data/scannet/feats_3d.pkl')

    with open(args.data_path, 'r') as f:
        datas = json.load(f)

    from base_grounder.configs.lmmgroundercfg import LMMGrounderConfig as cfg

    exp_name = "LMM:{}&vote_nums:{}&render_quality:{}&{}&".format(
        cfg.lmm_name, cfg.vote_nums, cfg.render_quality, "nr3d" if "nr3d" in args.data_path else "scanrefer")
    logger = comm.create_logger(exp_name + str(int(time.time())))
    logger.info("LMM: {} | vote_nums: {} | render_quality: {}".format(cfg.lmm_name, cfg.vote_nums, cfg.render_quality))
    cfg.log_self(logger)
    lmm_grounder = LMMGrounder(
        lmm=cfg.lmm_name, render_quality=cfg.render_quality,
        temperature=cfg.temperature,
        vote_nums=cfg.vote_nums, render_view_nums=cfg.render_view_nums
    )

    correct_25 = 0
    correct_50 = 0
    correct_easy = 0
    correct_dep = 0
    easy_total = 0
    dep_total = 0

    ground_errors = 0

    result = []
    datas = datas[:1000]

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
                index, pred_box = lmm_grounder.ask_lmm(sample)
                # _ = lmm_grounder.ground(sample)
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
                logger.info(f"Uid {sample['uid']}: {e}")
                ground_errors += 1
    finally:
        # json.dump(result, open(f'data/nr3d_val_{cfg.lmm_name}.json', 'w'), indent=4)
        logger.info('Easy: {}, Hard: {}, View-Dep: {}, View-Indep: {}'.format(
            correct_easy, correct_25 - correct_easy, correct_dep, correct_25 - correct_dep
        ))

        logger.info('Easy {} {} / {}'.format(correct_easy / easy_total, correct_easy, easy_total))
        logger.info('Hard {} {} / {}'.format((correct_25 - correct_easy) / (len(datas) - easy_total),
                                             correct_25 - correct_easy,
                                             len(datas) - easy_total))
        logger.info('View-Dep {} {} / {}'.format(correct_dep / dep_total, correct_dep, dep_total))
        logger.info('View-Indep {} {} / {}'.format((correct_25 - correct_dep) / (len(datas) - dep_total),
                                                   correct_25 - correct_dep,
                                                   len(datas) - dep_total))
        logger.info('Acc@25 {} {} / {}'.format(correct_25 / len(datas), correct_25, len(datas)))

        logger.info('Program Errors: {}'.format(ground_errors))
        logger.info('Programs errors rate: {}'.format(ground_errors / len(datas)))
