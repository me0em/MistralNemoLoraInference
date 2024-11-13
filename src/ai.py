import os

import yaml
from box import Box

import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
)

import numpy as np
from transformers import LlamaTokenizer
from transformers import pipeline
from peft import get_peft_model, PeftConfig, PeftModel


with open("config.yml", "r") as file:
    config = Box(yaml.safe_load(file))

os.environ["HF_TOKEN"] = config.hf_token

tokenizer = AutoTokenizer.from_pretrained(
    config.lora_path,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    token=config.hf_token,
    load_in_8bit=False,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

model.resize_token_embeddings(
    len(tokenizer),
    mean_resizing=False
)

model.tie_weights()
model = PeftModel.from_pretrained(model, config.lora_path)



def generate_text(model: AutoModelForCausalLM,
                  tokenizer: AutoTokenizer,
                  query: str,
                  device='cuda') -> str:
    """ Tokenize query, pass it to the model,
    convert tokens to human language text and return it
    """
    chat = [
        {"role": "user", "content": query},
    ]

    inputs = tokenizer.apply_chat_template(
        chat,
        return_tensors="pt",
        truncation=True,
        # padding="max_length",
        max_length=512,
        return_dict=True,
        add_generation_prompt=True
    ).to(device)
    
    # attention_mask = (
    #     inputs["input_ids"] != tokenizer.pad_token_id
    # ).long().to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            # attention_mask=attention_mask,
            # max_length=100,
            max_new_tokens=200,
            num_return_sequences=1,
            temperature=0.3
        )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    generated_text = generated_text[len(query):]    

    return generated_text


# model = lambda x: x
# tokenizer = lambda x: x
# generate_text = lambda model, tokenizer, query: query


if __name__ == "__main__":
    test_output = generate_text(
        model=model,
        tokenizer=tokenizer,
        query="Calc 10 + 5"
    )

    print(test_output)