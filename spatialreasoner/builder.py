"""
File: builder.py
Date: 2024/8/17
Author: yruns
"""


def build_pointcloud_tower(cfg):
    engine = cfg.pop("engine")
    if engine == "mask3d":
        from spatialreasoner.mask3d.builder import build_mask3d_segmentor
        return build_mask3d_segmentor(cfg)
    elif engine == "ptv3":
        from spatialreasoner.ptv3.point_transformerv3 import PointTransformerV3
        return PointTransformerV3(**cfg["model"])
    elif engine == "vote2cap":
        from spatialreasoner.vote2cap.builder import build_vote2cap
        return build_vote2cap(cfg)
    else:
        raise NotImplementedError(f"engine {cfg['engine']} is not implemented")


def build_grounding_tower(cfg):
    engine = cfg.pop("engine")
    if engine == "mask3d":
        from spatialreasoner.mask3d.builder import build_mask3d_segmentor
        return build_mask3d_segmentor(cfg)
    else:
        raise NotImplementedError(f"engine {cfg['engine']} is not implemented")
