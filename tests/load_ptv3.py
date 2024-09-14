# from spatialreasoner.ptv3.point_transformerv3 import PointTransformerV3
import torch
from loguru import logger
from trim.engine.default import load_state_dict
from trim.utils.comm import sum_model_parameters

if __name__ == '__main__':
    from configs.ptv3_conf import ptv3_cfg
    ptv3_cfg.pop("engine")

    state_path = "../pretrained/PTv3-Scannet200.pth"
    state = torch.load(state_path)

    from spatialreasoner.ptv3.point_transformerv3 import PointTransformerV3
    model = PointTransformerV3(**ptv3_cfg["model"])
    model = model.to(torch.bfloat16)
    # model.load_state_dict(state, strict=False)

    load_state_dict(state, model, logger, strict=False)

    print("done")