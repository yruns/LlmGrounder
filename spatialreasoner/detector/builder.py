from .config import DetectorConfig
from .criterion import build_criterion
from .detector import *


def build_vote2cap(scannet_config):
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
