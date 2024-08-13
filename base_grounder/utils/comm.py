"""
File: comm.py
Date: 2024/7/28
Author: yruns


"""
import base64
import logging
import os

import httpx
import numpy as np


def encode_image_to_base64(image_path):
    """Encodes an image to a base64 string."""

    if image_path.startswith('http'):
        return base64.b64encode(httpx.get(image_path).content).decode("utf-8")
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


def create_logger(exp_name):
    # Create a custom logger
    logger = logging.getLogger('my_logger')

    # Set the log level of logger to DEBUG
    logger.setLevel(logging.DEBUG)

    # Create handlers for writing to a file and logging to console
    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler(f'logs/{exp_name}.log', mode='w')
    console_handler = logging.StreamHandler()

    # Create formatters and add them to the handlers
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def calc_iou(box_a, box_b):
    """Computes IoU of two axis aligned bboxes.
    Args:
        box_a, box_b: 6D of center and lengths
    Returns:
        iou
    """

    max_a = box_a[0:3] + box_a[3:6] / 2
    max_b = box_b[0:3] + box_b[3:6] / 2
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3] - box_a[3:6] / 2
    min_b = box_b[0:3] - box_b[3:6] / 2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    return 1.0 * intersection / union
