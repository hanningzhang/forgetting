from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
from tqdm import tqdm

def apply_average(base_model_path, target_model_path, finetune_path):
    print(f"Loading the base weights from {base_model_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )

    print(f"Loading the finetune model from {finetune_path}")
    finetune = AutoModelForCausalLM.from_pretrained(
        finetune_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    merge_one = AutoModelForCausalLM.from_pretrained(
        finetune_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    merge_state_dict = {}
    print("Applying the Average")
    for name, param in tqdm(finetune.state_dict().items(), desc="Applying Average"):
        assert name in finetune.state_dict()
        #first = merge_one.state_dict()[name]
        merge_state_dict[name] = 0.7 * base.state_dict()[name] + 0.3 * finetune.state_dict()[name]
        #second = merge_one.state_dict()[name]
        #print(second-first)
    
    merge_one.load_state_dict(merge_state_dict)

    print(f"Saving the target model to {target_model_path}")
    merge_one.save_pretrained(target_model_path)
    base_tokenizer.save_pretrained(target_model_path)

base = "meta-llama/Meta-Llama-3.1-8B"
ft = "llama_5e-7_adam_1epoch_norm200"
target = "average_0.7pretrain_llama_5e-7_adam_1epoch_norm200"    
apply_average(base,target,ft)
