"""
File: test.py
Date: 2024/7/28
Author: yruns

Description: This file contains ...
"""

from utils.ground import PictureTaker

if __name__ == '__main__':
    picture_taker = PictureTaker("/data3/ysh/Datasets/ScanNet/scans")

    picture_taker.take_pictures("scene0000_00", 0, [])
