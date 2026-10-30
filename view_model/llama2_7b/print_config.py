# # 使用单张 CUDA GPU 和半精度，避免 7B 模型以 float32 加载后超出显存。
# import torch
# from transformers import pipeline
#
# pipe = pipeline(
#     "text-generation",
#     model="hf-models/Llama-2-7b-hf",
#     device=0,
#     dtype=torch.float16,
# )
from transformers import AutoConfig

model_path = "hf-models/Llama-2-7b-hf"
config = AutoConfig.from_pretrained(model_path)
print(config)
