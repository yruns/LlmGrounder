"""
File: llava.py
Date: 2024/7/29
Author: yruns


"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_ollama.llms import OllamaLLM

from base_grounder.utils import comm
from base_grounder.utils.ground import GrounderBase


class Llava(GrounderBase):

    def __init__(self, model="llava:34b", temperature: float = 0.8):
        self.llm = OllamaLLM(
            model=model, temperature=temperature
        )
        self.system_instruction = """You are a helpful image locator who can understand the text description I give you about a specific object. You need to answer which bounding box (bbox) in the image contains the target object from the text description. You only need to provide the index of the upper-left corner of the bbox and explain why you chose it. Wrap the output in `json` tags\n{format_instructions} \ndescription: {description}"""

    def invoke(self, description, images_path, candidate_bbox_nums):
        class Result(BaseModel):
            """
            The result of the image locator.
            """
            index: int = Field(
                default=None, enum=list(range(candidate_bbox_nums)),
                description="the index of the upper-left corner of the bbox")

            reason: str = Field(
                default=None,
                description="the reason why the index is chosen"
            )

        parser = PydanticOutputParser(pydantic_object=Result)
        prompt = PromptTemplate.from_template(self.system_instruction).partial(
            format_instructions=parser.get_format_instructions())

        llm_with_image_context = self.llm.bind(
            images=[comm.encode_image_to_base64(image_path) for image_path in images_path]
        )
        chain = prompt | llm_with_image_context | parser

        result = chain.invoke({"description": description})
        return result.index
