# import os

# import yaml
# from box import Box

# import torch
# from transformers import (
#     AutoModelForCausalLM,
#     BitsAndBytesConfig,
#     AutoTokenizer,
# )
# import numpy as np
# from transformers import LlamaTokenizer
# from transformers import pipeline
# from peft import get_peft_model, PeftConfig



# with open("config.yml", "r") as file:
#     config = Box(yaml.safe_load(file))

# os.environ["HF_TOKEN"] = config.open_ai_token

# tokenizer = AutoTokenizer.from_pretrained(
#     config.model_name,
#     trust_remote_code=True
# )

# tokenizer.pad_token = tokenizer.eos_token
# max_seq_length = tokenizer.model_max_length

# model = AutoModelForCausalLM.from_pretrained(
#     config.model_name,
#     token=config.hf_token,
#     load_in_8bit=False,
#     torch_dtype=torch.float16,
# )

# config = PeftConfig.from_pretrained(config.lora_path)

# model = get_peft_model(model, config)


# def generate_text(model: AutoModelForCausalLM,
#                   tokenizer: AutoTokenizer,
#                   query: str,
#                   device='cuda') -> str:
#     """ Tokenize query, pass it to the model,
#     convert tokens to human language text and return it
#     """
#     inputs = tokenizer(
#         query,
#         return_tensors="pt",
#         truncation=True,
#         padding="max_length",
#         max_length=512
#     ).to(device)
    
#     attention_mask = (inputs["input_ids"] != tokenizer.pad_token_id).long().to(device)

#     with torch.no_grad():
#         outputs = model.generate(
#             inputs["input_ids"],
#             attention_mask=attention_mask,
#             max_length=2048,
#             num_return_sequences=1
#         )

#     generated_text = tokenizer.decode(
#         outputs[0],
#         skip_special_tokens=True
#     )

#     return generated_text

model = lambda x: x
tokenizer = lambda x: x
generate_text = lambda model, tokenizer, query: query