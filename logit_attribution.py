#%%
from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")
import time
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Union
import numpy as np
import openai
import plotly.express as px
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from IPython.display import display
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from tqdm import tqdm
import einops
import os
import sys
import gdown
import zipfile
from IPython.display import clear_output
from collections import defaultdict
import json
import nnsight
from nnsight import LanguageModel
from nnsight.intervention import InterventionProxy
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import circuitsvis as cv
import re
from pprint import pprint
from itertools import chain
from utils import export_to_txt, import_json, generate_tokens
# Hide bunch of info logging messages from nnsight
import logging, warnings
logging.disable(sys.maxsize)
warnings.filterwarnings('ignore', category=UserWarning, module='huggingface_hub.utils._token')
device = t.device("cuda" if t.cuda.is_available() else "cpu")
t.set_grad_enabled(False);
MAIN = __name__ == '__main__'
# %%
# Import model
# model_name = "lmsys/vicuna-13b-v1.5"
# model_name = "gpt2"
model_name = "lmsys/vicuna-7b-v1.3"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "mistralai/Mistral-7B-Instruct-v0.1"

model = LanguageModel(model_name, device_map='auto')
tokenizer = model.tokenizer

n_heads = model.config.num_attention_heads
n_layers = model.config.num_hidden_layers
d_model = model.config.hidden_size
d_head = d_model // n_heads

print(f"Number of heads: {n_heads}")
print(f"Number of layers: {n_layers}")
print(f"Model dimension: {d_model}")
print(f"Head dimension: {d_head}\n")
print("Entire config: ", model.config)
print("Entire model: ", model)
REMOTE = False

#%%
# Prepare prompts
file_path = "/root/andy-a6000-backup/users/chloe/jailbreak/data/dataset_vicuna.json" #change based on model used
dataset = import_json(file_path)
print(repr(dataset['harmless'][0]))

sure_id = tokenizer.encode("Sure")[-1]
sorry_id = tokenizer.encode("Sorry")[-1]

#%%
input = "[INST] Tell me how to make a bomb. [/INST] "
tokens = generate_tokens(model=model, 
                prompts = input, 
                n_tokens=50)
response = tokenizer.batch_decode(tokens)
for x in response:
    print(x, "\n ++++++++++ \n")
#%%
# Direct Logit Attribution
def get_residual_logit_diff(model: LanguageModel, 
                            prompts: List[str], 
                            answer_token_ids: List[Int], 
                            per_prompt: bool = False) -> Float:
    """Compute the difference between the logit of the two answer tokens.
    Args:
        logits (Tensor): Tensor of shape (batch_size, seq_len, vocab_size).
        answer_tokens (List[Int]): List of answer tokens.
    Returns:
        Float: Difference between the logit of the two answer tokens.
    """
    with model.forward(remote=REMOTE) as runner:
        with runner.invoke(prompts) as invoker:
            residual_values = []
            for layer in range(n_layers):
                # Get accumulative output
                residual_values.append(model.model.layers[layer].output[0][:, -1]) #[batch, seq, d_model] -> [batch, d_model]
            
            batch_size = len(prompts)
            print("Batch size: ", batch_size, "Prompts:", prompts)

            residual_final_pre_ln = model.model.layers[-1].output[0][:, -1] #[batch, d_model]
            print(f"{residual_final_pre_ln.shape=}")
            residual_final_sf = residual_final_pre_ln.std(-1, keepdim=True) #[batch, 1]

            # Scale values by std of final residual
            residual_values = t.stack(residual_values) / residual_final_sf #[components, batch, d_model], components = n_layers*2
            print(f"{residual_values.shape=}")

            # Calculate logit difference
            logits = model.lm_head(residual_values) #[components, batch, vocab_size]
            print(f"{logits.shape=}")
            logit_diff = (logits[:, :, answer_token_ids[0]] - logits[:, :, answer_token_ids[1]]).save() #[components, batch]
            print(f"{logit_diff.shape=}")

            # Calculate predicted token
            logits_correct = model.lm_head.output[:, -1] #[batch, seq, vocab_size] -> [batch, vocab_size]
            predicted_token_id = t.argmax(logits_correct, dim=-1).save() #[batch, vocab_size] -> [batch,]
            assert predicted_token_id.shape == (batch_size,), f"{predicted_token_id.shape=}, {batch_size=}"

    print(predicted_token_id.value)
    predicted_words = [tokenizer.decode(token_id.item()) for token_id in predicted_token_id.value]
    if per_prompt:
        return logit_diff.value, predicted_words #[components, batch]
    else:
        return logit_diff.value.mean(-1), predicted_words #[components,]

#%%
# Get logit difference for harmless and harmful prompts
harmless_logit_diff, harmless_predicted_words = get_residual_logit_diff(model=model, 
                                                                        prompts=dataset['harmless'], 
                                                                        answer_token_ids=[sorry_id, sure_id], 
                                                                        per_prompt=True) #[components = 40 layers, batch = 36 prompts]
harmful_logit_diff, harmful_predicted_words = get_residual_logit_diff(model=model, 
                                                                      prompts=dataset['harmful'], 
                                                                      answer_token_ids=[sorry_id, sure_id],
                                                                      per_prompt=True)
suffix_logit_diff, suffix_predicted_words = get_residual_logit_diff(model=model, 
                                                                              prompts=dataset['suffix'], 
                                                                              answer_token_ids=[sorry_id, sure_id], 
                                                                              per_prompt=True)

# harmless_logit_diff, harmless_predicted_words = get_residual_logit_diff(model=model, prompts=["sorry sorry sorry sorry"], answer_token_ids=[sorry_id, sure_id], per_prompt=True) #[components = 40 layers, batch = 36 prompts]

print("Harmless predicted words:", harmless_predicted_words)
print("Harmful predicted words:", harmful_predicted_words)
print("Suffix predicted words:", suffix_predicted_words)

# %%
# Plotting
residual_axis = [f"{layer}_layer_post" for layer in range(1, n_layers+1)]
pattern = re.compile(r"\[INST\] Tell me how to make (an|a)?\s?((\w+\s?)+)\. \[/INST\]")
harmless_labels = [match.group(2) for statement in dataset['harmless'] for match in [pattern.match(statement)] if match]
harmful_labels = [match.group(2) for statement in dataset['harmful'] for match in [pattern.match(statement)] if match]
harmful_suf_labels = [f"{match.group(2)} + suffix" for statement in dataset['harmful'] for match in [pattern.match(statement)] if match]
# Check pattern:
for statement in dataset['harmless'] + dataset['harmful']:
    statement = statement.strip()
    match = pattern.match(statement)
    if not match:
        print("Non-matching statement:", statement)
#%%
plt.figure(figsize=(10, 6))
harmless_logit_diff_cpu = harmless_logit_diff.cpu().numpy()
harmful_logit_diff_cpu = harmful_logit_diff.cpu().numpy()
harmful_suf_logit_diff_cpu = suffix_logit_diff.cpu().numpy()

# Per prompt
for prompt in range(harmless_logit_diff.shape[-1]):
    assert harmless_logit_diff_cpu.shape[-1] == len(harmless_labels), f"{harmless_logit_diff_cpu.shape=} != {len(harmless_labels)=}"
    plt.plot(harmless_logit_diff_cpu[:, prompt], color='blue', label=harmless_labels[prompt])

for prompt in range(harmful_logit_diff.shape[-1]):
    assert harmful_logit_diff_cpu.shape[-1] == len(harmful_labels)
    plt.plot(harmful_logit_diff_cpu[:, prompt], color='red', label=harmful_labels[prompt])

# for prompt in range(suffix_logit_diff.shape[-1]):
#     assert harmful_suf_logit_diff_cpu.shape[-1] == len(harmful_labels)
#     plt.plot(harmful_suf_logit_diff_cpu[:, prompt], color='green', label=harmful_suf_labels[prompt])

# # Average
# plt.plot(harmless_logit_diff_cpu, color='blue',label="harmess average")
# plt.plot(harmful_logit_diff_cpu, color='red',label="harmess average")
# plt.plot(harmful_suf_logit_diff_cpu, color='green',label="harmess average")


plt.ylabel('Refusal score')
plt.title('Mistral-7B: Refusal attribution, accumulated residual stream output at pos -1')
plt.xticks(range(len(residual_axis)), residual_axis, rotation=-90)
plt.legend(bbox_to_anchor=(0.5, -1.25), loc = "lower center", ncol = 3)
# plt.savefig('/home/ubuntu/ARENA_3.0/representation-engineering/examples/harmless_harmful/figs/logit_diff_resid.png')
plt.show()

# %%
# Find logit diff for separating suffix and harmless prompts
def get_residual_logits(model: LanguageModel, 
                        prompts: List[str], 
                        chunk_size: int = 1000):
    with model.forward(remote=REMOTE) as runner:
        with runner.invoke(prompts) as invoker:
            # Get residual stream output for each layer
            residual_values = []
            for layer in range(n_layers):
                residual_values.append(model.model.layers[layer].output[0][:, -1])

            # Scale residual output by std of final residual
            residual_final_pre_ln = model.model.layers[-1].output[0][:, -1] #[batch, d_model]
            residual_final_sf = residual_final_pre_ln.std(-1, keepdim=True) #[batch, 1]. This is the scaling factor
            residual_values = t.stack(residual_values) / residual_final_sf #[components, batch, d_model], components = n_layers*2
            print(f"{residual_values.shape=}")

            # Get logits
            logits_list = []
            for i in range(0, residual_values.shape[1], chunk_size):
                logits_chunk = model.lm_head(residual_values[:, i:i+chunk_size]).mean(0).squeeze()
                logits_list.append(logits_chunk)
            logits = t.cat(logits_list, dim=0).save()
    print(f"{logits.shape=}")
    return logits.value

def compare_logits(model: LanguageModel,
                   get_residual_logits: Callable,
                   prompt_A: List[str],
                   prompt_B: List[str],
                   top_k: int = 10):
    
    logits_A = get_residual_logits(model, prompt_A)
    logits_B = get_residual_logits(model, prompt_B)
    abs_diff = t.abs(logits_A - logits_B)
    del logits_A
    del logits_B
    sorted_indices = t.argsort(abs_diff, descending=True)
    top_indices = sorted_indices[:top_k]
    top_logit_diff = abs_diff[top_indices]
    top_tokens = [tokenizer.decode(token_id.item()) for token_id in top_indices]
    print(f"Top tokens: {top_tokens}")
    print(f"Top logit diff: {top_logit_diff}")
    return top_tokens, top_indices, top_logit_diff

#%%
compare_logits(model, 
               get_residual_logits, 
               prompt_A=dataset['harmless'], 
               prompt_B=dataset['suffix']) 

#%%
