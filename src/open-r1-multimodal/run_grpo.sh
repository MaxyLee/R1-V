dt=$(date '+%d_%m_%Y_%H:%M_%S');
export DEBUG_MODE="true"
export LOG_PATH="/home/maxinyu/logs/grpo_aircraft_7b.txt"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# CUDA_VISIBLE_DEVICES=7

CKPT=/home/maxinyu/ckpt/Qwen2-VL-7B-Instruct
OUTPUT_DIR=/home/maxinyu/exp/R1-V/Qwen2-VL-7B-Instruct/test

DATASET=/home/maxinyu/data/oven/grounding/aircraft/concat-random.parquet

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
    --num_train_epochs 2 \
    --run_name Qwen2-VL-7B-aircraft-grounding \
    --save_steps 1000 \
    --save_only_model true \
    --deepspeed local_scripts/zero3.json

# python src/open_r1/grpo.py \
#     --output_dir $OUTPUT_DIR \
#     --model_name_or_path $CKPT \
#     --dataset_name $DATASET \
#     --max_prompt_length 1024 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 2 \
#     --logging_steps 1 \
#     --bf16 \
#     --report_to wandb \
#     --gradient_checkpointing false \
#     --attn_implementation flash_attention_2 \
#     --max_pixels 401408 \
#     --max_completion_length 1024 \
#     --num_train_epochs 1 \
#     --run_name Qwen2-VL-7B-aircraft-grounding \
#     --save_steps 1000 \
#     --save_only_model true