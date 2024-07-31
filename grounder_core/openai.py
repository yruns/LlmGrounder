"""
File: openai.py
Date: 2024/7/29
Author: yruns

Description: This file contains ...
"""
import os

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from utils import comm
from utils.ground import GrounderBase

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_API_KEY"] = "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"


class GPT(GrounderBase):

    def __init__(self, model, api_key, base_url):
        self.model = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url
        )
        self.system_instruction = """You are a helpful image locator who can understand the text descriptions I give you about a specific object. You need to answer which bounding box (bbox) in the image contains the target object from the text description. You only need to provide the index of the upper-left corner of the bbox and explain why you chose it. "Wrap the output in `json` tags\n{format_instructions}"
        """

    def invoke(self, description, images_path, candidate_bbox_nums):
        content = [{"type": "text", "text": description}]
        for i in range(len(images_path)):
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{comm.encode_image_to_base64(images_path[i])}"
            }})
        message = [
            ("system", self.system_instruction),
            HumanMessage(content=content)
        ]

        class Result(BaseModel):
            """
            The result of the image locator.
            """
            index: int = Field(
                default=None, enum=range(len(images_path)),
                description="the index of the upper-left corner of the bbox")

            reason: str = Field(
                default=None,
                description="the reason why the index is chosen"
            )

        parser = PydanticOutputParser(pydantic_object=Result)

        prompt = ChatPromptTemplate.from_messages(message).partial(
            format_instructions=parser.get_format_instructions())
        chain = prompt | self.model | parser

        result = chain.invoke({})
        return result.index
