from .segmentor import Mask3DSegmentor

def build_mask3d_segmentor(cfg):
    return Mask3DSegmentor(**cfg)