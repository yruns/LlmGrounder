"""
File: prompts.py
Date: 2024/8/16
Author: yruns
"""
from .const import REF_TOKEN

SYSTEM_PROMPTS = [
    # "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    "You are an advanced AI model tasked with identifying and segmenting objects in 3D scenes based on natural language descriptions. Your primary goal is to accurately output the segment mask of the target object mentioned in the description provided to you."
]

ASK_PROMPTS = {
    "seg": [
        "{SCENE_TOKEN}\nGiven the 3D scene, can you help ground the object by its description which is {UTTERANCE}?",
        "{SCENE_TOKEN}\nBased on the provided 3D scene, locate the object described as {UTTERANCE} and generate its segment mask.",
        "{SCENE_TOKEN}\nIn the given 3D scene, identify and segment the object that matches the following description: {UTTERANCE}.",
        "{SCENE_TOKEN}\nCan you segment the object in this 3D scene that corresponds to the description: {UTTERANCE}?",
        "{SCENE_TOKEN}\nPlease find and segment the object mentioned in the description: {UTTERANCE}, using the provided 3D scene.",
        "{SCENE_TOKEN}\nUsing the 3D scene, can you identify and output the segment mask for the object described as {UTTERANCE}?",
        "{SCENE_TOKEN}\nRefer to the 3D scene and provide the segment mask for the object that fits the following description: {UTTERANCE}.",
        "{SCENE_TOKEN}\nIn the given 3D environment, segment the object that aligns with this description: {UTTERANCE}.",
        "{SCENE_TOKEN}\nUsing the 3D scene, generate the segment mask for the object matching this description: {UTTERANCE}.",
        "{SCENE_TOKEN}\nFrom the 3D scene, identify the object that corresponds to {UTTERANCE} and output its segment mask.",
        "{SCENE_TOKEN}\nBased on the 3D scene, segment the object that is described as {UTTERANCE} and provide its mask.",
        "{SCENE_TOKEN}\nCan you find the object described in {UTTERANCE} within the 3D scene and output its segment mask?",
        "{SCENE_TOKEN}\nBased on the description {UTTERANCE}, segment the corresponding object in the 3D scene.",
        "{SCENE_TOKEN}\nGiven the 3D scene, please locate and segment the object described as {UTTERANCE}.",
        "{SCENE_TOKEN}\nIn the 3D scene, identify and provide the segmentation for the object described by {UTTERANCE}.",
        # "{SCENE_TOKEN}\nUsing the 3D scene as a reference, segment the object that matches the description {UTTERANCE}.",
        # "{SCENE_TOKEN}\nRefer to the 3D scene and extract the segment mask for the object mentioned in {UTTERANCE}.",
        # "{SCENE_TOKEN}\nFrom the 3D scene, can you find and segment the object described as {UTTERANCE}?",
        # "{SCENE_TOKEN}\nUsing the description {UTTERANCE}, locate the corresponding object in the 3D scene and produce its segment mask.",
        # "{SCENE_TOKEN}\nIdentify the object mentioned in {UTTERANCE} from the 3D scene and generate its segmentation mask.",
        # "{SCENE_TOKEN}\nGiven the 3D scene, please highlight the object that corresponds to the description: {UTTERANCE} and output its segment mask.",
    ]
}

REPLY_PROMPTS = {
    "seg": [
        f"Sure, it is {REF_TOKEN}.",
        f"It is {REF_TOKEN}.",
        f"Sure, {REF_TOKEN}.",
        f"Sure, it is {REF_TOKEN}.",
        f"Sure, the segmentation result is {REF_TOKEN}.",
        f"{REF_TOKEN}.",
    ]
}
