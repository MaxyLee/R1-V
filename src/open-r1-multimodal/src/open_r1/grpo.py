# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import torch
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk, Image
from transformers import Qwen2VLForConditionalGeneration
from torchvision.ops.boxes import box_area

# from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

ds_collections = {
    'aircraft': '/home/maxinyu/data/oven/grounding/aircraft/concat-random-r1.parquet',
    'aircraft-f': '/home/maxinyu/data/oven/grounding/aircraft/concat-random-r1-filtered.parquet',
    'aircraft-f2': '/home/maxinyu/data/oven/grounding/aircraft/concat-random-r1-filtered2.parquet',
    'car': '/home/maxinyu/data/oven/grounding/car/concat-random-r1.parquet',
    'dog': '/home/maxinyu/data/oven/grounding/dog/concat-random-r1.parquet',
    'reptilia': '/home/maxinyu/data/oven/grounding/reptilia/concat-random-r1.parquet',
    'bird': '/home/maxinyu/data/oven/grounding/bird/concat-random-r1.parquet',
    'food': '/home/maxinyu/data/oven/grounding/food/concat-random-r1.parquet',
}

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'iou', 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["iou", "format"],
        metadata={"help": "List of reward functions. Possible values: 'iou', 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    
answer_pattern = re.compile(r'<answer>(.*?)</answer>')
bbox_patterns = [
    re.compile(r'\((\d*?),.*?(\d*?)\),\((\d*?),(\d*?)\)'),
    re.compile(r'\[(\d*?), (\d*?), (\d*?), (\d*?)\]'),
    re.compile(r'\((\d*?), (\d*?), (\d*?), (\d*?)\)'),
    re.compile(r'\((\d*?), (\d*?)\)\n?.*?\((\d*?), (\d*?)\)'),
]
sensation_pattern = re.compile(r'<sensation>(.*?)</sensation>')
sensation_bbox_pattern = re.compile(r'\((\d*?),.*?(\d*?)\),\((\d*?),(\d*?)\)')
bbox_threshold = 0.75

stop_words = ['top left', 'top right', 'bottom left', 'bottom right', ]

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def get_bbox(ans):
    for i, pattern in enumerate(bbox_patterns):
        predict_bbox = re.findall(pattern, ans)
        if len(predict_bbox) != 0:
            try:
                predict_bbox = (float(predict_bbox[-1][0].replace('[', '').replace('x', '')), float(predict_bbox[-1][1]), float(predict_bbox[-1][2]), float(predict_bbox[-1][3]))
            except:
                predict_bbox = [0, 0, 0, 0]
            # if sum(predict_bbox) < 4:
            #     predict_bbox = [c*1000 for c in predict_bbox]
            return predict_bbox, i+1
    
    return (0., 0., 0., 0.), 0

def get_sensation_bbox(sen):
    predict_bbox = re.findall(sensation_pattern, sen)
    if len(predict_bbox) != 0:
        try:
            predict_bbox = (float(predict_bbox[-1][0].replace('[', '').replace('x', '')), float(predict_bbox[-1][1]), float(predict_bbox[-1][2]), float(predict_bbox[-1][3]))
        except:
            predict_bbox = [0, 0, 0, 0]
        return predict_bbox
    
    return (0., 0., 0., 0.)

def iou_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
            
        gt_bbox, _ = get_bbox(sol)
        gt_bbox = torch.tensor(gt_bbox, dtype=torch.float32).view(-1, 4)
        answer = re.findall(answer_pattern, content)
        if len(answer) > 0:
            pred_bbox, _ = get_bbox(answer[0])
            pred_bbox = torch.tensor(pred_bbox, dtype=torch.float32).view(-1, 4)
            iou, _ = box_iou(gt_bbox, pred_bbox)
            iou = iou.item()
            reward = iou if iou > bbox_threshold else 0.0

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} IoU reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards

def sensation_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    gt_bboxes = kwargs['sensation']
    for content, gt_bbox in zip(contents, gt_bboxes):
        reward = 0.0

        gt_bbox = torch.tensor(gt_bbox, dtype=torch.float32).view(-1, 4)
        
        match = re.match(sensation_pattern, content)
        # sensation = 
        pred_bbox, _ = get_bbox(content)
        pred_bbox = torch.tensor(pred_bbox, dtype=torch.float32).view(-1, 4)
        iou, _ = box_iou(gt_bbox, pred_bbox)
        iou = iou.item()
        reward = iou if iou > 0.75 else 0.0

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} IoU reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        # try:
        #     answer = parse(content)
        #     if float(verify(answer, parse(sol))) > 0:
        #         reward = 1.0
        # except Exception:
        #     pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
            except Exception as e:
                print(e)
                print(f"------------- {current_time} Accuracy reward: {reward} -------------")
                print(f"Content: {content}")
                print(f"Solution: {sol}")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>[\S\n\t\v ]*?</think>\s*<answer>.*?</answer>"
    # pattern = r"<sensation>.*?</sensation><think>[\S\n\t\v ]*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    answer_format_reward = [0.5 if match else 0.0 for match in matches]
    
    bbox_format_reward = []
    for content in completion_contents:
        answer = re.findall(answer_pattern, content)
        if len(answer) > 0:
            _, match_type = get_bbox(answer[0])
            if match_type == 1:
                bbox_format_reward.append(0.5)
            else:
                bbox_format_reward.append(0.0)
        else:
            bbox_format_reward.append(0.0)
            
    reward = [a+b for a,b in zip(answer_format_reward, bbox_format_reward)]
    return reward

def bbox_format_reward(completions, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    
    bbox_format_reward = []
    for content in completion_contents:
        _, match_type = get_bbox(content)
        if match_type == 1:
            bbox_format_reward.append(1.0)
        else:
            bbox_format_reward.append(0.0)
        # else:
        #     bbox_format_reward.append(0.1)
            
    return bbox_format_reward

reward_funcs_registry = {
    "sensation": sensation_reward,
    "iou": iou_reward,
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first briefly describes the visual information that helps answer the question"
    "then thinks about the reasoning process in the mind and then provides the user with the answer. The visual information, reasoning "
    "process, and answer are enclosed within <semsation> <sensation>, <think> </think>, and <answer> </answer> tags, respectively, i.e., "
    "<sensation> visual information here </sensation><think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    log_path = os.getenv("LOG_PATH")
    if not os.path.isfile(log_path):
        with open(log_path, 'w') as f:
            f.write('\n')
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    # import ipdb; ipdb.set_trace()
    # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset_names = script_args.dataset_name.split(',')
    dataset_files = [ds_collections[ds_name] for ds_name in dataset_names]
    dataset = load_dataset('parquet', data_files=dataset_files).cast_column("image", Image())

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (bounding box) in <answer> </answer> tags."

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }

    R1S_QUESTION_TEMPLATE = "{Question}  Output the sensation process in <sensation> </sensation>, the thinking process in <think> </think>, and final answer in <answer> </answer> tags."
    
    def make_conversation_image_sensation(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": R1S_QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }

    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        if "sensation" in dataset[script_args.dataset_train_split].features:
            print("has sensation in dataset")
            dataset = dataset.map(make_conversation_image_sensation)  # Utilize multiprocessing for faster mapping
        else:
            dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    
    trainer_cls = Qwen2VLGRPOTrainer
    

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
