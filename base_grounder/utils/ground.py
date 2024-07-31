"""
File: ground.py
Date: 2024/7/28
Author: yruns

Description: This file contains ...
"""
import cv2
import glob
import logging
import numpy as np
import open3d as o3d
import os
import pickle
import torch
import torch.nn as nn
import utils.render as render_utils
from PIL import Image
from transformers import AutoTokenizer, CLIPModel
from typing import List, Dict, Union
from typing import Literal

from data.scannet200_constants import CLASS_LABELS_200

feats: Union[List[Dict], None] = None


def lazy_load_feats(feat_path="data/scannet/feats_3d.pkl"):
    global feats
    if feats is None:
        with open(feat_path, 'rb') as f:
            feats = pickle.load(f)


def load_pc(scan_id):
    obj_ids = feats[scan_id]['obj_ids']
    inst_locs = feats[scan_id]['inst_locs']
    center = feats[scan_id]['center']
    obj_embeds = feats[scan_id]['obj_embeds']

    return obj_ids, inst_locs, center, obj_embeds


class PictureTaker(object):

    def __init__(self, scan_root, view_nums=4, quality: Literal["low", "high"] = "low"):
        super().__init__()
        self.scan_root = scan_root
        self.view_nums = view_nums
        self.quality = quality

    def take_pictures(self, scan_id, uid, bboxes):
        ply_type = "{}_vh_clean{}.ply".format(scan_id, "" if self.quality == "high" else "_2")
        ply_file = os.path.join(self.scan_root, scan_id, ply_type)
        pcd: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(str(ply_file))

        # 将mesh按照axis_align_matrix进行变换
        axis_align_matrix = render_utils.read_axis_align_matrix(
            os.path.join(os.path.dirname(ply_file), f"{scan_id}.txt")
        )
        mesh_vertices = np.asarray(pcd.vertices)
        aligned_vertices = render_utils.align_point_vertices(mesh_vertices, axis_align_matrix)
        pcd.vertices = o3d.utility.Vector3dVector(aligned_vertices)

        bboxes = [
            render_utils.Convertor.convert_bbox_to_o3d_format(bbox, center_type=True)
            for bbox in bboxes
        ]

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, height=1080, width=1920)
        vis.add_geometry(pcd)

        center = pcd.get_center()
        max_bound = pcd.get_max_bound()
        min_bound = pcd.get_min_bound()

        scene_circle_r = max(np.max(np.abs(min_bound - center)[:2]), np.max(np.abs(max_bound - center)[:2]))

        store_dir = f"rendered_views/{scan_id}/{uid}"
        if self.quality == "high":
            store_dir = os.path.join(store_dir, "high")
        os.makedirs(store_dir, exist_ok=True)
        rendered_views_path = []

        for view_idx in range(self.view_nums):
            camera_lookat = pcd.get_center()
            z_height = 3
            angle = view_idx * 2 * np.pi / self.view_nums
            camera_pos = np.array([scene_circle_r * np.sin(angle), scene_circle_r * np.cos(angle), z_height])

            rendered_view = self.render_view(camera_pos, camera_lookat, bboxes=bboxes, vis=vis, store_dir=store_dir,
                                             view_idx=view_idx)

            rendered_view_path = os.path.join(store_dir, f"view_{view_idx}.png")
            cv2.imwrite(rendered_view_path, rendered_view)
            rendered_views_path.append(rendered_view_path)

        return rendered_views_path

    @staticmethod
    def render_view(camera_pos, camera_lookat, bboxes, vis, store_dir, view_idx):
        # 设置摄像头参数
        ctr: o3d.visualization.ViewControl = vis.get_view_control()

        front_vector = camera_pos - camera_lookat
        up = render_utils.compute_up_vector(front_vector)

        ctr.set_up(up)
        ctr.set_front(front_vector)
        ctr.set_lookat(camera_lookat)
        ctr.set_zoom(0.45)

        # 设置渲染选项
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.9, 0.9, 0.9])  # 设置背景颜色
        vis.poll_events()
        vis.update_renderer()

        # 捕捉图像
        image = vis.capture_screen_float_buffer(do_render=True)
        image = render_utils.Convertor.convert_o3d_image_to_cv2_image(image)

        rendered_original_view_path = os.path.join(store_dir, f"original_view_{view_idx}.png")
        cv2.imwrite(rendered_original_view_path, image)

        bbox_corners_2d = render_utils.project_bbox_with_pinhole_camera_parameters(
            camera_params=ctr.convert_to_pinhole_camera_parameters(),
            bboxes=bboxes, return_wherther_in_view=True
        )

        # 将2D BBox画在图像上
        for idx, (top_left, bottom_right, is_in_view) in enumerate(bbox_corners_2d):
            image = cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), thickness=3)

            # 添加order
            image = render_utils.add_order_to_image(
                image=image, order=idx, top_left=top_left, size=(30, 40), return_rgb=False
            )

        return image


class GrounderBase(object):

    def invoke(self, description, images_path):
        pass


class Locator(object):

    def __init__(self, tokenizer_name='clip-vit-base-patch16'):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=512, use_fast=True)
        self.clip = CLIPModel.from_pretrained(tokenizer_name).cuda()

        self.class_name_list = list(CLASS_LABELS_200)
        self.class_name_list.remove('wall')
        self.class_name_list.remove('floor')
        self.class_name_list.remove('ceiling')

        self.class_name_tokens = self.tokenizer([f'a {class_name} in a scene' for class_name in self.class_name_list],
                                                padding=True,
                                                return_tensors='pt')
        for name in self.class_name_tokens.data:
            self.class_name_tokens.data[name] = self.class_name_tokens.data[name].cuda()

        label_lang_infos = self.clip.get_text_features(**self.class_name_tokens)
        self.label_lang_infos = label_lang_infos / label_lang_infos.norm(p=2, dim=-1, keepdim=True)

    def locate(self, obj_name, scan_id):
        obj_ids, inst_locs, center, obj_embeds = load_pc(scan_id)

        boxes = []
        # cosine similarity as logits
        class_logits_3d = torch.matmul(self.label_lang_infos, obj_embeds.t().cuda())  # * logit_scale
        obj_cls = class_logits_3d.argmax(dim=0)
        pred_class_list = [self.class_name_list[idx] for idx in obj_cls]

        new_class_list = list(set(pred_class_list))
        new_class_name_tokens = self.tokenizer([f'a {class_name} in a scene' for class_name in new_class_list],
                                               padding=True,
                                               return_tensors='pt')

        for name in new_class_name_tokens.data:
            new_class_name_tokens.data[name] = new_class_name_tokens.data[name].cuda()
        label_lang_infos = self.clip.get_text_features(**new_class_name_tokens)
        label_lang_infos = label_lang_infos / label_lang_infos.norm(p=2, dim=-1, keepdim=True)

        query_name_tokens = self.tokenizer([f'a {obj_name} in a scene'], padding=True, return_tensors='pt')
        for name in query_name_tokens.data:
            query_name_tokens.data[name] = query_name_tokens.data[name].cuda()

        query_lang_infos = self.clip.get_text_features(**query_name_tokens)
        query_lang_infos = query_lang_infos / query_lang_infos.norm(p=2, dim=-1, keepdim=True)

        text_cls = torch.matmul(query_lang_infos, label_lang_infos.t())
        text_cls = text_cls.argmax(dim=-1)[0]
        text_cls = new_class_list[text_cls]

        for i in range(len(obj_ids)):
            if pred_class_list[i] == text_cls:
                boxes.append(inst_locs[i])

        return boxes
