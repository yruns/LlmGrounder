import statistics

import MinkowskiEngine as ME

from .metrics import IoU

from trim.utils import comm


class Mask3DSegmentor(object):

    def __init__(self, config):
        self.config = config

        self.decoder_id = config["general"]["decoder_id"]
        if config["model"]["train_on_segments"]:
            self.mask_type = "segment_mask"
        else:
            self.mask_type = "masks"

        ## => Mask3D Model
        from .mask3d import Mask3D
        self.model = Mask3D(**config["model"])

        ## => Loss
        self.ignore_label = config["data"]["ignore_label"]
        from .matcher import HungarianMatcher
        matcher = HungarianMatcher(**config["matcher"])
        weight_dict = {
            "loss_ce": matcher.cost_class,
            "loss_mask": matcher.cost_mask,
            "loss_dice": matcher.cost_dice,
        }
        aux_weight_dict = {}
        for i in range(self.model.num_levels * self.model.num_decoders):
            if i not in self.config.general.ignore_mask_idx:
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in weight_dict.items()}
                )
            else:
                aux_weight_dict.update(
                    {k + f"_{i}": 0.0 for k, v in weight_dict.items()}
                )
        weight_dict.update(aux_weight_dict)

        self.preds = dict()
        self.bbox_preds = dict()
        self.bbox_gt = dict()

        from .criterion import SetCriterion
        self.criterion = SetCriterion(
            **config["loss"], matcher=matcher, weight_dict=weight_dict
        )

        ## => metrics
        from .metrics import ConfusionMatrix
        self.confusion = ConfusionMatrix(**config["metrics"])
        self.iou = IoU()

        ## => misc
        self.labels_info = dict()


    def encode_scene(self, batch, batch_idx, is_eval):
        data, target, file_names = batch

        if data.features.shape[0] > self.config.general.max_batch_size:
            print("data exceeds threshold")
            raise RuntimeError("BATCH TOO BIG")

        if len(target) == 0:
            print("no targets")
            return None

        raw_coordinates = None
        if self.config.data.add_raw_coordinates:
            raw_coordinates = data.features[:, -3:]
            data.features = data.features[:, :-3]

        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=data.device,
        )

        encoder_out, mask_features = self.model.encode_scene(
            data,
            point2segment=[
                target[i]["point2segment"] for i in range(len(target))
            ],
            raw_coordinates=raw_coordinates,
            is_eval=is_eval,
        )
        return encoder_out, batch, mask_features


    def decode(self, encoder_out, batch):
        data, target, file_names = batch
        x, point2segment, is_eval, aux, coordinates, \
            pcd_features, batch_size, coords, pos_encodings_pcd, mask_features = encoder_out

        output = self.model.decode(x, point2segment, is_eval, aux, coordinates,
            pcd_features, batch_size, coords, pos_encodings_pcd, mask_features)

        if not is_eval:
            ## => Training, compute loss
            try:
                losses = self.criterion(output, target, mask_type=self.mask_type)
            except ValueError as val_err:
                print(f"ValueError: {val_err}")
                print(f"data shape: {data.shape}")
                print(f"data feat shape:  {data.features.shape}")
                print(f"data feat nans:   {data.features.isnan().sum()}")
                print(f"output: {output}")
                print(f"target: {target}")
                print(f"filenames: {file_names}")
                raise val_err

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return sum(losses.values()) / len(losses)

        else:
            ## => Evaluation, compute metrics
            return None

