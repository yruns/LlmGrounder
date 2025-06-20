from datasets.scannet import DatasetConfig


class DetectorConfig:

    def __init__(self, mask3d_cfg):
        self.dataset_config = DatasetConfig()
        self.num_class = self.dataset_config.num_semcls

        # preencoder: Set Abstraction Layer
        self.in_channel = (
                3 * (int(mask3d_cfg["use_color"]) + int(mask3d_cfg["use_normal"])) +
                1 * int(mask3d_cfg["use_height"]) +
                128 * int(mask3d_cfg["use_multiview"])
        )
        self.preenc_npoints = 2048

        # position embedding
        self.pos_embed = 'fourier'

        # encoder
        self.enc_type = 'masked'
        self.enc_nlayers = 3
        self.enc_dim = 256
        self.enc_ffn_dim = 128
        self.enc_dropout = 0.1
        self.enc_nhead = 4
        self.enc_activation = 'relu'

        # decoder
        self.nqueries = 256
        self.dec_nlayers = 8
        self.dec_dim = 256
        self.dec_ffn_dim = 256
        self.dec_dropout = 0.1
        self.dec_nhead = 4

        # mlp heads
        self.mlp_dropout = 0.3

        ### Matcher
        self.matcher_giou_cost = 2.
        self.matcher_cls_cost = 1.
        self.matcher_center_cost = 0.
        self.matcher_objectness_cost = 0.

        ### Loss Weights
        self.loss_giou_weight = 10.
        self.loss_sem_cls_weight = 1.
        self.loss_no_object_weight = 0.25
        self.loss_angle_cls_weight = 0.1
        self.loss_angle_reg_weight = 0.5
        self.loss_center_weight = 5.
        self.loss_size_weight = 1.
