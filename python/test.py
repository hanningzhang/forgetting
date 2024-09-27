from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import transformers
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from data import JsonDataset, tokenize_prompt, tokenize_conversion, tokenize_text_only, tokenize_conversion_lmflow
from prompt_maker import PromptMaker
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import logging.config
import os
import sys
from tqdm import tqdm
import wandb
from functools import partial
from parse_args import parse_argument, parse_args_dict
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, Qwen2ForCausalLM, Gemma2ForCausalLM
from CustomModels.modeling_llama import LlamaForCausalLM
from CustomModels.modeling_qwen2 import Qwen2ForCausalLM
from CustomModels.modeling_gemma2 import Gemma2ForCausalLM

from torch import linalg as LA
# from NormModel import ModelWithLPNorm
from utils import get_optimizer, evaluate_and_logging, make_tqdm, save_model, logging_stat_dict
import time
import shutil

model_path = "../llama_2e-7_adam_1epoch_norm0"
tokenizer=transformers.AutoTokenizer.from_pretrained(model_path)
model = transformers.AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16).eval().to(0)

if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

transform=partial(tokenize_conversion_lmflow, response_loss_only=True, max_length=1024, tokenizer=tokenizer)

dataset=JsonDataset(json_data="../data_lmflow/test.json", 
                                 shuffle=True, train=True,
                                 transform=transform,
                                 chunk_long_text=False,
                                 lmflow_format=True)

val_loader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=transformers.DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=1, return_tensors="pt"),
        num_workers=1,
        batch_size=1)


model.eval()
sum_loss = 0.

for batch in tqdm(val_loader):
    batch=batch.to(0)
    #print(batch)
    x_batch=batch['input_ids']
    y_batch=batch['labels']
    attn_mask=batch['attention_mask']
    x_batch = x_batch# .cuda()
    y_batch = y_batch# .cuda()
    attn_mask = attn_mask# .cuda()

        # if args.norm:
            # loss, norm, res = model(input_ids=x_batch, labels=y_batch, attention_mask=attn_mask, return_dict=True)
        # else:
    with torch.no_grad():    
        loss = model(input_ids=x_batch, labels=y_batch, attention_mask=attn_mask, return_dict=True).loss

        #sum_loss += accelerator.gather(loss).detach().cpu().mean()
        #sum_loss += loss.detach().cpu().mean()
        sum_loss += loss

num_batch = len(val_loader)
eval_loss = sum_loss / float(num_batch)
print(f"total loss: {sum_loss}")
print(f"evaluation loss: {eval_loss}")
