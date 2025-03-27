dt=$(date '+%d_%m_%Y_%H:%M_%S');
export DEBUG_MODE="true"
RUN_NAME=oven-all-cot-grpo
# RUN_NAME=oven-aircraft-grpo
# RUN_NAME=foci-all-grpo
# RUN_NAME=test
export LOG_PATH="/home/maxinyu/logs/$RUN_NAME($dt).txt"

# oven-aircraft-grpo
# lr 1e-6 iou threshold 0.8 w/o cot-sft

# oven-aircraft-grpo (Qwen2.5VL)
# lr 1e-6 iou threshold 0.5 w/o cot-sft

# oven-aircraft-cot-grpo-6
# lr 1e-6 iou threshold 0.8(?)

# oven-aircraft-cot-grpo-7
# lr 1e-6 iou threshold 0.5

# oven-aircraft-cot-grpo-8
# lr 1e-6 iou threshold 0.9

# oven-aircraft-cot-grpo-9
# lr 2e-6 iou threshold 0.8

# oven-aircraft-cot-grpo-10
# lr 1e-6 iou threshold 0.85 beta 0.06

# oven-all-cot-grpo
# lr 1e-6 iou threshold 0.85 beta 0.04 (default)

# oven-all-cot-grpo-2
# lr 3e-6 iou threshold 0.8

# oven-all-cot-grpo-3
# lr 1e-6 iou threshold 0.85 beta 0.02

# oven-all-cot-grpo-4
# lr 1e-6 iou threshold 0.85, data r1-3

# oven-all-cot-grpo-5
# lr 1e-6 iou threshold 0.8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# CUDA_VISIBLE_DEVICES=0

# CKPT=/home/maxinyu/ckpt/Qwen2-VL-7B-Instruct
# CKPT=/home/maxinyu/ckpt/Qwen2.5-VL-7B-Instruct

# CKPT=/home/maxinyu/exp/Qwen2-VL/Qwen2-VL-7B-Instruct/oven-aircraft-cot-r1
# CKPT=/home/maxinyu/exp/Qwen2-VL/Qwen2-VL-7B-Instruct/oven-aircraft-cot-r1s-2
# CKPT=/home/maxinyu/exp/Qwen2-VL/Qwen2-VL-7B-Instruct/oven-aircraft-cot-r1-slerp
# CKPT=/home/maxinyu/exp/Qwen2-VL/Qwen2-VL-7B-Instruct/oven-all3-cot-r1
CKPT=/home/maxinyu/exp/Qwen2-VL/Qwen2.5-VL-7B-Instruct/oven-all-cot-r1
OUTPUT_DIR=/home/maxinyu/exp/R1-V/Qwen2.5-VL-7B-Instruct/$RUN_NAME

# DATASET=aircraft
# DATASET=aircraft-f3
# DATASET=aircraft,car,dog,repptilia,bird,food
# DATASET=aircraft-f3,car-f3,reptilia-f3,bird-f3,food-f3
DATASET=aircraft-f3-qwen2.5,car-f3-qwen2.5,reptilia-f3-qwen2.5,bird-f3-qwen2.5,food-f3-qwen2.5
# DATASET=aircraft-f4,car-f4,reptilia-f4,bird-f4,food-f4
# DATASET=aircraft-f3s,car-f3s,reptilia-f3s,bird-f3s,food-f3s
# DATASET=foci-dog,foci-bird,foci-aircraft,foci-flower,foci-pet,foci-car
# DATASET=foci-aircraft


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

# vllm
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node="7" \
#     --nnodes="1" \
#     --node_rank="0" \
#     --master_addr="127.0.0.1" \
#     --master_port="12543" \
#     src/open_r1/grpo.py --use_vllm True \
#     --output_dir $OUTPUT_DIR \
#     --model_name_or_path $CKPT \
#     --dataset_name $DATASET \
#     --max_prompt_length 512 \
#     --max_completion_length 512 \
#     --learning_rate 1.0e-6 \
#     --temperature 1.0 \
#     --num_generations 4 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --logging_steps 1 \
#     --bf16  \
#     --report_to wandb \
#     --gradient_checkpointing true \
#     --attn_implementation flash_attention_2 \
#     --max_pixels 400000 \
#     --max_steps 13125 \
#     --run_name $RUN_NAME \
#     --save_steps 100 \
#     --save_only_model true \
#     --deepspeed local_scripts/zero3.json 

# debug
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