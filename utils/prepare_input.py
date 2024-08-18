"""
File: prepare_input.py
Date: 2024/8/17
Author: yruns
"""
import random

from fontTools.ttLib.tables.ttProgram import instructions

from staticvars.prompts import *
from staticvars.const import *
from utils.tokenize import tokenize_scene_token

def assemble_instruction(utterance, granularity):
    """
    Assemble instruction
    :param utterance
    :param granularity: seg or reg
    :return:
    """
    instruction_snippet_list = []

    # Add system prompt
    system_prompt = random.choice(SYSTEM_PROMPTS)
    instruction_snippet_list.append(system_prompt)

    # Add ask prompt
    ask_role = ROLES["ask"]
    ask_prompt = random.choice(ASK_PROMPTS[granularity]).format(SCENE_TOKEN=SCENE_TOKEN, UTTERACNE=utterance)
    instruction_snippet_list.append("{role}: {prompt}".format(role=ask_role, prompt=ask_prompt))

    # Add reply prompt
    reply_role = ROLES["reply"]
    reply_prompt = random.choice(REPLY_PROMPTS[granularity])
    instruction_snippet_list.append(
        "{role}: {prompt}{end_token}".format(role=reply_role, prompt=reply_prompt, end_token=REPLY_END_TOKEN)
    )

    return "".join(instruction_snippet_list)    # Modifying sep may cause errors in `tokenize_scene_token` function


def prepare_for_llm(
    instruction,
    tokenizer,
    has_scene=True
):
    """
    1. Tokenize the instruction;
    2. Make a deepcopy as the target. Mask reply words with IGNORE_INDEX.
    """
    input_ids = tokenize_scene_token(
        instruction,
        tokenizer,
        scene_token=SCENE_TOKEN,
        scene_token_index=SCENE_TOKEN_INDEX
    )

    target_ids = input_ids.clone()



    return input_ids, target_ids
