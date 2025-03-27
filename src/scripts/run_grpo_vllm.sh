#!/bin/bash

# The latest vllm==0.7.2 is required for this script: pip3 install vllm==0.7.2 

RUN_NAME=oven-aircraft-cot-grpo-6

export DEBUG_MODE="true"
export LOG_PATH="/home/maxinyu/logs/$RUN_NAME($dt).txt"

QWEN_PATH=/home/maxinyu/exp/Qwen2-VL/Qwen2-VL-7B-Instruct/oven-aircraft-cot-r1
DATASET=aircraft-f
OUTPUT_DIR=/home/maxinyu/exp/R1-V/Qwen2-VL-7B-Instruct/$RUN_NAME

# NOTE: you are expected to use X + 1 cards for X training proc and 1 vLLM proc 
# e.g., the visible devices should be 0,1,2,3,4 for 5 cards, and  --nproc_per_node="4"

CUDA_VISIBLE_DEVICES="3,4,5,6,7" torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12543" \
    src/open_r1/grpo.py --use_vllm True \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $QWEN_PATH \
    --dataset_name $DATASET \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --learning_rate 1.0e-6 \
    --temperature 1.0 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16  \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 400000 \
    --max_steps 13125 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --deepspeed local_scripts/zero3.json 
