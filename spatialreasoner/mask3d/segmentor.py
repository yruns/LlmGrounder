import MinkowskiEngine as ME
import numpy as np
import json
from pathlib import Path
import torch
from torch import nn
from torch_scatter import scatter_mean

from .metrics import IoU


class Mask3DSegmentor(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.device = None
        self.config = config

        self.eval_on_segments = config.general.eval_on_segments

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

        num_labels = config["data"]["train_dataset"]["num_labels"]
        labels = json.load(open(Path(config["data"]["train_dataset"]["label_db_filepath"]), "r"))
        labels = {int(key): value for key, value in labels.items() if isinstance(key, str)}
        # if working only on classes for validation - discard others
        _labels = self._select_correct_labels(labels, num_labels)
        ## => misc
        self.labels_info = _labels

    @staticmethod
    def _select_correct_labels(labels, num_labels):
        number_of_validation_labels = 0
        number_of_all_labels = 0
        for (
                k,
                v,
        ) in labels.items():
            number_of_all_labels += 1
            if v["validation"]:
                number_of_validation_labels += 1

        if num_labels == number_of_all_labels:
            return labels
        elif num_labels == number_of_validation_labels:
            valid_labels = dict()
            for (
                    k,
                    v,
            ) in labels.items():
                if v["validation"]:
                    valid_labels.update({k: v})
            return valid_labels
        else:
            msg = f"""not available number labels, select from:
                {number_of_validation_labels}, {number_of_all_labels}"""
            raise ValueError(msg)

    def encode(self, batch, is_eval, device):
        data, target, file_names = batch
        self.device = device

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
            device=device,
        )

        encoder_state = self.model.encode(
            data,
            point2segment=[
                target[i]["point2segment"] for i in range(len(target))
            ],
            raw_coordinates=raw_coordinates,
            is_eval=is_eval,
        )
        return encoder_state

    def decode(self, encoder_state, batch, queries_pos, is_eval=False):
        data, target, file_names = batch
        point2segment, is_eval, aux, pcd_features, \
            coords, pos_encodings_pcd, mask_features, _, _ = encoder_state

        queries = torch.zeros_like(queries_pos).permute(1, 0, 2)
        output = self.model.decode(point2segment, is_eval, aux, pcd_features,
                                   coords, pos_encodings_pcd, mask_features, queries, queries_pos)

        try:
            losses = self.criterion(output, target, mask_type=self.mask_type)
        except ValueError as val_err:
            raise val_err

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        loss = sum(losses.values()) / len(losses)
        if not is_eval:
            ## => Training, compute loss
            return loss

        else:
            ## => Evaluation, compute metrics
            inverse_maps = data.inverse_maps
            target_full = data.target_full
            original_coordinates = data.original_coordinates

            raw_coordinates = None
            if self.config.data.add_raw_coordinates:
                raw_coordinates = data.features[:, -3:]
                data.features = data.features[:, :-3]

            pred_classes, pred_masks, pred_scores, pred_bbox_data, gt_bbox_data = self.eval_instance_step(
                output,
                target,
                target_full,
                inverse_maps,
                original_coordinates,
                raw_coordinates,
                backbone_features=None,
            )
            return dict(
                loss=loss,
                pred_classes=pred_classes,
                pred_masks=pred_masks,
                pred_scores=pred_scores,
                pred_bboxes=pred_bbox_data,
                gt_bboxes=gt_bbox_data,
            )

    def get_mask_and_scores(
            self, mask_cls, mask_pred, num_queries=100, num_classes=18, device=None
    ):
        if device is None:
            device = self.device
        labels = (
            torch.arange(num_classes, device=device)
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )

        if self.config.general.topk_per_image != -1:
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                self.config.general.topk_per_image, sorted=True
            )
        else:
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                num_queries, sorted=True
            )

        labels_per_query = labels[topk_indices]
        topk_indices = topk_indices // num_classes
        mask_pred = mask_pred[:, topk_indices]

        result_pred_mask = (mask_pred > 0).float()
        heatmap = mask_pred.float().sigmoid()

        mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (
                result_pred_mask.sum(0) + 1e-6
        )
        score = scores_per_query * mask_scores_per_image
        classes = labels_per_query

        return score, result_pred_mask, classes, heatmap

    def get_full_res_mask(
            self, mask, inverse_map, point2segment_full, is_heatmap=False
    ):
        mask = mask.detach().cpu()[inverse_map]  # full res

        if self.eval_on_segments and is_heatmap == False:
            mask = scatter_mean(
                mask, point2segment_full, dim=0
            )  # full res segments
            mask = (mask > 0.5).float()
            mask = mask.detach().cpu()[
                point2segment_full.cpu()
            ]  # full res points

        return mask

    def remap_model_output(self, output):
        output = np.array(output)
        output_remapped = output.copy()
        for i, k in enumerate(self.labels_info.keys()):
            output_remapped[output == i] = k
        return output_remapped

    def eval_instance_step(
            self,
            output,
            target_low_res,
            target_full_res,
            inverse_maps,
            full_res_coords,
            raw_coords,
            first_full_res=False,
            backbone_features=None,
    ):
        label_offset = 2  # `from mask3d_conf.py`
        prediction = output["aux_outputs"]
        prediction.append({
            "pred_logits": output["pred_logits"],
            "pred_masks": output["pred_masks"],
        })

        prediction[self.decoder_id][
            "pred_logits"
        ] = torch.functional.F.softmax(
            prediction[self.decoder_id]["pred_logits"], dim=-1
        )[..., :-1]

        all_pred_classes = list()
        all_pred_masks = list()
        all_pred_scores = list()
        all_heatmaps = list()

        offset_coords_idx = 0
        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            if not first_full_res:
                if self.model.train_on_segments:
                    masks = (
                        prediction[self.decoder_id]["pred_masks"][bid]
                        .detach()
                        .cpu()[target_low_res[bid]["point2segment"].cpu()]
                    )
                else:
                    masks = (
                        prediction[self.decoder_id]["pred_masks"][bid]
                        .detach()
                        .cpu()
                    )

                if self.config.general.use_dbscan:
                    new_preds = {
                        "pred_masks": list(),
                        "pred_logits": list(),
                    }

                    curr_coords_idx = masks.shape[0]
                    curr_coords = raw_coords[offset_coords_idx: curr_coords_idx + offset_coords_idx]
                    offset_coords_idx += curr_coords_idx

                    for curr_query in range(masks.shape[1]):
                        curr_masks = masks[:, curr_query] > 0

                        if curr_coords[curr_masks].shape[0] > 0:
                            clusters = (
                                DBSCAN(
                                    eps=self.config.general.dbscan_eps,
                                    min_samples=self.config.general.dbscan_min_points,
                                    n_jobs=-1,
                                )
                                .fit(curr_coords[curr_masks])
                                .labels_
                            )

                            new_mask = torch.zeros(curr_masks.shape, dtype=int)
                            new_mask[curr_masks] = (
                                    torch.from_numpy(clusters) + 1
                            )

                            for cluster_id in np.unique(clusters):
                                original_pred_masks = masks[:, curr_query]
                                if cluster_id != -1:
                                    new_preds["pred_masks"].append(
                                        original_pred_masks * (new_mask == cluster_id + 1)
                                    )
                                    new_preds["pred_logits"].append(
                                        prediction[self.decoder_id][
                                            "pred_logits"
                                        ][bid, curr_query]
                                    )

                    scores, masks, classes, heatmap = self.get_mask_and_scores(
                        torch.stack(new_preds["pred_logits"]).cpu(),
                        torch.stack(new_preds["pred_masks"]).T,
                        len(new_preds["pred_logits"]),
                        self.model.num_classes - 1,
                    )
                else:
                    scores, masks, classes, heatmap = self.get_mask_and_scores(
                        prediction[self.decoder_id]["pred_logits"][bid]
                        .detach()
                        .cpu(),
                        masks,
                        prediction[self.decoder_id]["pred_logits"][bid].shape[0],
                        self.model.num_classes - 1,
                    )

                masks = self.get_full_res_mask(
                    masks,
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                )

                heatmap = self.get_full_res_mask(
                    heatmap,
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                    is_heatmap=True,
                )

                if backbone_features is not None:
                    backbone_features = self.get_full_res_mask(
                        torch.from_numpy(backbone_features),
                        inverse_maps[bid],
                        target_full_res[bid]["point2segment"],
                        is_heatmap=True,
                    )
                    backbone_features = backbone_features.numpy()
            else:
                assert False, "not tested"
                masks = self.get_full_res_mask(
                    prediction[self.decoder_id]["pred_masks"][bid].cpu(),
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                )

                scores, masks, classes, heatmap = self.get_mask_and_scores(
                    prediction[self.decoder_id]["pred_logits"][bid].cpu(),
                    masks,
                    prediction[self.decoder_id]["pred_logits"][bid].shape[0],
                    self.model.num_classes - 1,
                    device="cpu",
                )

            masks = masks.numpy()
            heatmap = heatmap.numpy()

            sort_scores = scores.sort(descending=True)
            sort_scores_index = sort_scores.indices.cpu().numpy()
            sort_scores_values = sort_scores.values.cpu().numpy()
            sort_classes = classes[sort_scores_index]

            sorted_masks = masks[:, sort_scores_index]
            sorted_heatmap = heatmap[:, sort_scores_index]

            if self.config.general.filter_out_instances:
                keep_instances = set()
                pairwise_overlap = sorted_masks.T @ sorted_masks
                normalization = pairwise_overlap.max(axis=0)
                norm_overlaps = pairwise_overlap / normalization

                for instance_id in range(norm_overlaps.shape[0]):
                    # filter out unlikely masks and nearly empty masks
                    # if not(sort_scores_values[instance_id] < 0.3 or sorted_masks[:, instance_id].sum() < 500):
                    if not (
                            sort_scores_values[instance_id]
                            < self.config.general.scores_threshold
                    ):
                        # check if mask != empty
                        if not sorted_masks[:, instance_id].sum() == 0.0:
                            overlap_ids = set(
                                np.nonzero(
                                    norm_overlaps[instance_id, :]
                                    > self.config.general.iou_threshold
                                )[0]
                            )

                            if len(overlap_ids) == 0:
                                keep_instances.add(instance_id)
                            else:
                                if instance_id == min(overlap_ids):
                                    keep_instances.add(instance_id)

                keep_instances = sorted(list(keep_instances))
                all_pred_classes.append(sort_classes[keep_instances])
                all_pred_masks.append(sorted_masks[:, keep_instances])
                all_pred_scores.append(sort_scores_values[keep_instances])
                all_heatmaps.append(sorted_heatmap[:, keep_instances])
            else:
                all_pred_classes.append(sort_classes)
                all_pred_masks.append(sorted_masks)
                all_pred_scores.append(sort_scores_values)
                all_heatmaps.append(sorted_heatmap)

        # if self.validation_dataset.dataset_name == "scannet200":
        all_pred_classes[bid][all_pred_classes[bid] == 0] = -1
        # if self.config.data.test_mode != "test":
        target_full_res[bid]["labels"][
            target_full_res[bid]["labels"] == 0
            ] = -1

        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            all_pred_classes[bid] = self.remap_model_output(
                all_pred_classes[bid].cpu() + label_offset
            )

            if (
                    self.config.data.test_mode != "test"
                    and len(target_full_res) != 0
            ):
                target_full_res[bid]["labels"] = self.remap_model_output(
                    target_full_res[bid]["labels"].cpu() + label_offset
                )

                # PREDICTION BOX
                pred_bbox_data = []
                for query_id in range(
                        all_pred_masks[bid].shape[1]
                ):  # self.model.num_queries
                    obj_coords = full_res_coords[bid][
                                 all_pred_masks[bid][:, query_id].astype(bool), :
                                 ]
                    if obj_coords.shape[0] > 0:
                        obj_center = obj_coords.mean(axis=0)
                        obj_axis_length = obj_coords.max(
                            axis=0
                        ) - obj_coords.min(axis=0)

                        bbox = np.concatenate((obj_center, obj_axis_length))

                        pred_bbox_data.append(
                            (
                                all_pred_classes[bid][query_id].item(),
                                bbox,
                                all_pred_scores[bid][query_id],
                            )
                        )

                # GT BOX
                gt_bbox_data = []
                for obj_id in range(target_full_res[bid]["masks"].shape[0]):
                    if target_full_res[bid]["labels"][obj_id].item() == 255:
                        continue

                    obj_coords = full_res_coords[bid][
                                 target_full_res[bid]["masks"][obj_id, :]
                                 .cpu()
                                 .detach()
                                 .numpy()
                                 .astype(bool),
                                 :,
                                 ]
                    if obj_coords.shape[0] > 0:
                        obj_center = obj_coords.mean(axis=0)
                        obj_axis_length = obj_coords.max(
                            axis=0
                        ) - obj_coords.min(axis=0)

                        bbox = np.concatenate((obj_center, obj_axis_length))
                        gt_bbox_data.append(
                            (
                                target_full_res[bid]["labels"][obj_id].item(),
                                bbox,
                            )
                        )

        if len(pred_bbox_data) == 0:
            pred_bbox_data = [(-1, np.zeros(6), 0)]
        # We only return the first one, because we only give one query at a time during inference
        return all_pred_classes[0], all_pred_masks[0], all_pred_scores[0], pred_bbox_data[0], gt_bbox_data[0]
