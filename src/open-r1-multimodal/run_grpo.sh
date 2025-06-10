dt=$(date '+%d_%m_%Y_%H:%M_%S');
export DEBUG_MODE="true"
RUN_NAME=kvg-grpo

export LOG_PATH="$RUN_NAME($dt).txt"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CKPT=path/to/stage1-checkpoint
DATASET=aircraft,car,reptilia,bird,food
OUTPUT_DIR=path/to/output_dir

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $CKPT \
    --dataset_name $DATASET \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --max_completion_length 512 \
    --learning_rate 1.0e-6 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-7B-$RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --deepspeed local_scripts/zero3.json \
    --num_generations 4
