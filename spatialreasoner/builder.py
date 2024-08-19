"""
File: builder.py
Date: 2024/8/17
Author: yruns
"""
from .detector.detector import *
from .detector.criterion import build_criterion
from .detector.config import DetectorConfig

def build_mm_detector(scannet_config):
    cfg = DetectorConfig(scannet_config)

    tokenizer = build_preencoder(cfg)
    encoder = build_encoder(cfg)
    decoder = build_decoder(cfg)

    criterion = build_criterion(cfg, cfg.dataset_config)

    model = Vote2CapDETR(
        tokenizer,
        encoder,
        decoder,
        cfg.dataset_config,
        encoder_dim=cfg.enc_dim,
        decoder_dim=cfg.dec_dim,
        mlp_dropout=cfg.mlp_dropout,
        num_queries=cfg.nqueries,
        criterion=criterion
    )
    return model, cfg


def build_mm_segmentor():
    pass
