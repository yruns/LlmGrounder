"""
File: builder.py
Date: 2024/8/17
Author: yruns
"""

def build_pointcloud_tower(cfg):
    if cfg["engine"] == "mask3d":
        from spatialreasoner.mask3d.builder import build_mask3d_segmentor
        return build_mask3d_segmentor(cfg)
    elif cfg["engine"] == "vote2cap":
        from spatialreasoner.detector.builder import build_vote2cap
        return build_vote2cap(cfg)
    else:
        raise NotImplementedError(f"engine {cfg['engine']} is not implemented")


def build_mm_segmentor():
    pass
