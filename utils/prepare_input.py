"""
File: prepare_input.py
Date: 2024/8/17
Author: yruns
"""
import random

from staticvars.const import *
from staticvars.prompts import *


def assemble_instruction(utterance, granularity):
    """
    Assemble instruction
    :param utterance
    :param granularity: seg or reg
    :return:
    """
    instruction_snippet_list = []

    ### => Add system prompt
    system_prompt = random.choice(SYSTEM_PROMPTS)
    instruction_snippet_list.append(system_prompt)

    ### => Add ask prompt
    ask_role = ROLES["ask"]
    ask_prompt = random.choice(ASK_PROMPTS[granularity]).format(SCENE_TOKEN=SCENE_TOKEN, UTTERANCE=utterance)
    instruction_snippet_list.append("{role}: {prompt}".format(role=ask_role, prompt=ask_prompt))

    ### => Add reply prompt
    reply_role = ROLES["reply"]
    reply_prompt = random.choice(REPLY_PROMPTS[granularity])
    instruction_snippet_list.append(
        "{role}: {prompt}{end_token}".format(role=reply_role, prompt=reply_prompt, end_token=REPLY_END_TOKEN)
    )

    return "\n".join(instruction_snippet_list)
