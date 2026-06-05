#!/bin/bash

train_ds="../../data/ds_train"
valid_ds="../../data/ds_valid"
attributes_map="../../data/attributes_map.json"
celltype_embed="../../data/stack-embed.npy"
output_dir="../../biolord_logs_stack"


DATAPARAMS="
    --train_ds=$train_ds \
    --valid_ds=$valid_ds \
    --attributes_map=$attributes_map \
    --celltype_embed=$celltype_embed"


TRAINPARAMS="
    --seed=42 \
    --output_dir=$output_dir \
    --num_train_epochs=200 \
    --logging_steps=50 \
    --checkpointing_steps=epoch-20 \
    --eval_every_n_epoch=1 \
    --earlystop_patience=20 \
    --per_device_train_batch_size=512 \
    --per_device_eval_batch_size=512 \
    --gradient_accumulation_steps=1 \
    --max_grad_norm=1.0 \
    --learning_rate=1e-4 \
    --lr_scheduler_type=cosine \
    --weight_decay=1e-4 \
    --num_warmup_ratio=0 \
    --mixed_precision=bf16 \
    --with_tracking=True \
    --report_to=tensorboard \
    --dataloader_pin_memory=True \
    --dataloader_persistent_workers=True \
    --dataloader_num_workers=16 \
    --dataloader_prefetch_factor=2 \
    --alpha_mse_loss=10000 \
    --unknown_attribute_penalty=0.1"


export CUDA_VISIBLE_DEVICES="0"
accelerate launch \
    --config_file="./accelerate_config.yaml" \
    --num_processes=1 \
    train.py \
    $TRAINPARAMS \
    $DATAPARAMS