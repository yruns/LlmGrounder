"""
File: qwen.py
Date: 2024/7/31
Author: yruns


"""
import json

import dashscope

from base_grounder.utils.ground import GrounderBase


class Qwen(GrounderBase):

    def __init__(self, model, api_key):
        dashscope.api_key = api_key
        self.model = model
        self.system_instruction = """You are a helpful image locator who can understand the text description I give you about a specific object. You need to answer which bounding box (bbox) in the image contains the target object from the text description. You only need to provide the index of the upper-left corner of the bbox and explain why you chose it. "Wrap the output in `json` tags\nThe output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"description": "The result of the image locator.", "properties": {"index": {"title": "Index", "description": "the index of the upper-left corner of the bbox", "type": "integer"}, "reason": {"title": "Reason", "description": "the reason why the index is chosen", "type": "string"}}}\n```"""

    def invoke(self, description, images_path, candidate_bbox_nums):
        messages = [
            {
                "role": "user",
                "content": [{"text": self.system_instruction}]
            },
            {
                "role": "user",
                "content": [{"image": f"file://{path}"} for path in images_path] +
                           [{"text": f"description: {description}"}]

            }
        ]
        response = dashscope.MultiModalConversation.call(model=self.model, messages=messages)
        result = response.output.choices[0].message.content[0]["text"]

        # remove json tags
        try:
            result = json.loads(result)
        except:
            result = json.loads(result.replace("```json", "").replace("```", ""))

        return int(result["index"])
