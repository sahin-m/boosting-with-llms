#Copyright 2026 Mücahit Sahin
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.


import numpy as np
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    pipeline
)


def load(repo="Qwen/Qwen3.5-9B", quantization=None, context_length=None):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(repo)
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Avoid warning

    # Load config
    config = AutoConfig.from_pretrained(repo)
    
    if context_length is not None:
        config.max_position_embeddings = context_length
        config.initial_context_length = context_length

    # Determine kwargs for model loading
    model_kwargs = {
        "device_map": "auto",
        "config": config
    }

    # Use Hugging Face's built-in quantization options
    if quantization == "4bit":
        model_kwargs["load_in_4bit"] = True
        model_kwargs["torch_dtype"] = torch.bfloat16  # recommended for 4bit
    elif quantization == "8bit":
        model_kwargs["load_in_8bit"] = True
    # else: let HF handle pre-quantized models automatically

    # Load model
    model = AutoModelForCausalLM.from_pretrained(repo, **model_kwargs) #, offload_folder="offload")

    # Create generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    return pipe



def ask(prompt, system_prompt=None, pipe=None):
    if pipe is None:
        return "Model was not loaded! Use load_model() before running this function."

    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [
            {"role": "user", "content": prompt},
        ]

    try:
        # Use chat template + disable thinking
        formatted_prompt = pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        outputs = pipe(
            formatted_prompt,
            max_new_tokens=5,
            max_length=None,
            do_sample=False,
            temperature=0.02,
            eos_token_id=pipe.tokenizer.eos_token_id
        )

        response = outputs[0]["generated_text"]

    finally:
        torch.cuda.empty_cache()

    return response