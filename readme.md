# LLM Forgetting

## Install
```bash
pip install -U -r requirements.txt
```

## Run
```bash
./train.sh
```

## Arguments
```
--model             model name or path (transformer compatible)
--model-type        model type (current support Llama, Gemma2, Qwen2)
--tokenizer-name    model name or path (transformer compatible)
--train-data        can use wildcard for multiple files in a dir
--val-data          can use wildcard for multiple files in a dir
--optimizer         arguments passed to optimizer
--norm 0.1          if specified, will use this value as lambda for l2 penalty. Otherwise no penalty.
--bf16              default mode. Do not modify unless necessary
--pseudo_random     fix random value generator
--logging_conf_file conf/common.log_conf default mode. Do not modify unless necessary
--seed              random seed
--max-steps         if specified, will ignore dataset size and use this value as max optimization steps
--diff_norm         if turned on, will use $\Vert\theta-\theta_0\Vert_2^2$ as penalty
--val_batch_size    validation batch size
--eval_frequency    evaluate on val data every k steps
--save_dir          dir to save model
--sharegpt_format   turn on if using chat data e.g. data/val.json
--lmflow-format     turn on if using lmflow format data e.g. data_lmflow/*
--max-length        the maximum input token length
--global_batch_size
--response_loss_only 
--micro_batch_size 
```

## Notes
- note there are different scripts to run baseline/norm and diff_norm. This is because of different fsdp config
- to run diff_norm, use GPU with memory larger than 50 GB
- uses could set --norm to 0 to log the param norm during training
- do not turn on cpu offload
- if change model, also change fsdp_transformer_layer_cls_to_wrap in fsdp_config.yaml (GPT2Block for gpt2 and LlamaDecoderLayer for Llama)
- default lr warmup ratio 0.03
- check data loading [here](python/data.py#L216)
- if using wandb, export WANDB_API_KEY and set args --use_wandb, --wandb_project, --wandb_run_name. refer parse_args.py