"""
File: test.py
Date: 2024/7/28
Author: yruns

Description: This file contains ...
"""

import base64
from io import BytesIO


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def convert_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG", optimize=False)  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


if __name__ == '__main__':
    import time

    print(time.time())
    print("aa" + str(int(time.time())))
