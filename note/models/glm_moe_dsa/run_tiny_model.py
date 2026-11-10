import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("inference-optimization/GLM-5.3-0.6B-A0.4B")
model = AutoModelForCausalLM.from_pretrained(
    "inference-optimization/GLM-5.3-0.6B-A0.4B", dtype="auto"
).to("cuda:0" if torch.cuda.is_available() else "cpu")
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
