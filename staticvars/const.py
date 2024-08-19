"""
File: const.py
Date: 2024/8/16
Author: yruns
"""

SCENE_TOKEN = "<scene/>"
SCENE_TOKEN_INDEX = -201
SCENE_START_TOKEN = "<scene_start/>"
SCENE_END_TOKEN = "<scene_end/>"

REPLY_END_TOKEN = "</s>"

IGNORE_INDEX = -100  # Which is default ignored by nn.CrossEntropyLoss()

TASK_TOKEN = "<task>"
REG_TOKEN = "<reg>"
SEG_TOKEN = "<seg>"

# ROLES = {"ask": "USER", "reply": "ASSISTANT"}
ROLES = {"ask": "## human", "reply": "## assistant"}
