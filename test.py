# from transformers import AutoTokenizer,AutoModelForCausalLM
from datasets import load_dataset, Dataset
import torch
from vllm import LLM, SamplingParams

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
from transformers import pipeline


tokenizer = AutoTokenizer.from_pretrained('Qwen_2e-6_adam_1epoch_diffnorm25')
model = AutoModelForCausalLM.from_pretrained('Qwen_2e-6_adam_1epoch_diffnorm25').eval()