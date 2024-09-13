"""
File: const.py
Date: 2024/8/16
Author: yruns
"""

MAX_NEW_TOKENS = 15

IGNORE_INDEX = -100  # Which is default ignored by nn.CrossEntropyLoss()

SCENE_TOKEN = "<scene/>"
SCENE_TOKEN_INDEX = -201
SCENE_START_TOKEN = "<scene_start>"
SCENE_END_TOKEN = "<scene_end>"


REF_TOKEN = "[ref]"
SOP_TOKEN = "<p>"
EOP_TOKEN = "</p>"
REPLY_END_TOKEN = "</s>"

ROLES = {"ask": "USER", "reply": "ASSISTANT"}
# ROLES = {"ask": "## human", "reply": "## assistant"}
