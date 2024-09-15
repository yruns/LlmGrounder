import torch

from trim.engine.default import load_state_dict
from trim.thirdparty.logging import logger

if __name__ == '__main__':
    from configs.mask3d_conf import mask3d_cfg

    mask3d_cfg.pop("engine")

    from trim.utils.config import ConfigDict

    mask3d_cfg = ConfigDict(mask3d_cfg)

    state_path = "../pretrained/Mask3D-Scannet200.pth"
    state = torch.load(state_path)

    from spatialreasoner.mask3d.builder import build_mask3d_segmentor

    model = build_mask3d_segmentor(mask3d_cfg)
    # model.load_state_dict(state, strict=False)
    load_state_dict(state, model, logger, strict=False)

    print("done")
