import copy

import transformers
import torch


model_id = "pretrained/vicuna-7b-v1.3"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

import time
start = time.time()
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
end = time.time()
print(end - start)