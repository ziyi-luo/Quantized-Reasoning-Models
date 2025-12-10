import os
import json
import random
import argparse
from tqdm import tqdm

import torch
import torch_npu
import transformers
from vllm import LLM
from vllm.engine.arg_utils import PoolerConfig


# 给 npu 加的，清理临时存储的，但暂时没用上
import gc 
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel)

from lighteval.models.model_input import GenerationParameters
from lighteval_custom.models.vllm.vllm_model import VLLMModelConfig
from lighteval_custom.main_vllm import vllm
from vllm_custom.model_executor.fake_quantized_models.registry import register_fake_quantized_models
register_fake_quantized_models()    # register fake-quantized models in vLLM


def parser_gen():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="whether to re-evaluate")
    parser.add_argument('--load_responses_from_json_file', type=str, default=None,
                        help='Load response from json file.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to save inference results.')
    # model
    parser.add_argument('--model', type=str, default='./modelzoo/Meta-Llama-3-8B',
                        help='Model to load.')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='dtype to use')
    # dataset
    parser.add_argument('--dataset', type=str, default='ARC-E',
                        choices=["ARC-E", "ARC-C", "HellaSwag", "LAMBADA", "PIQA", "WinoGrande"],
                        help='Dataset to load.')
    parser.add_argument('--max_samples', type=int, default=None, help='Max #samples (for debug)')
    # generation
    parser.add_argument('--temperature', type=float, default=0.6, help='Generation temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Generation top_p')
    parser.add_argument('--seed', type=int, default=42, help='Generation seed')
    parser.add_argument('--max_new_tokens', type=int, default=32768,
                        help='Maximum number of tokens to generate per output sequence.')
    # 修改最大模型长度，来适配llama
    parser.add_argument('--max_model_length', type=int, default=8192,
                        help='Maximum model input length.')
    args = parser.parse_args()

    # force float16 for gptqmodel inference
    if "gptqmodel" in args.model:
        args.dtype = "float16"

    # output path
    args.model_name = args.model.split("/")[-1]
    output_dir = os.path.join("./outputs", "inference", f"{args.model_name}-seed{args.seed}") if args.output_dir is None else args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    args.output_path = os.path.join(output_dir, f"{args.dataset}.jsonl")

    # Distributed settings
    # args.tensor_parallel_size = torch.cuda.device_count()
    args.tensor_parallel_size = torch.npu.device_count()
    # args.tensor_parallel_size = 4

    return args


# 给 npu 加的，清理临时存储的，但暂时没用上
def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()
    


def main(args):
    if not args.debug and not args.overwrite and os.path.exists(args.output_path):
        print(f"Evaluation results found at {args.output_path}. Skip evaluation")
        return

    random.seed(args.seed)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    if args.debug:
        import debugpy
        debugpy.listen(5678)
        args.max_new_tokens = 10
        args.max_samples = 2

    generation_parameters = GenerationParameters(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=30 if "QwQ" in args.model else None,  # TODO. enable top_k only for QwQ?
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        # # 给4bit加一个惩罚重复生成的参数
        # repetition_penalty = 1.05,
    )
    model_config = VLLMModelConfig(
        pretrained=args.model,
        dtype=args.dtype,
        max_model_length=args.max_model_length,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        generation_parameters=generation_parameters,
        init_model=(args.load_responses_from_json_file is None),
    )

    if args.dataset == "ARC-E":
        task_kwargs = {
            "tasks": "custom|arc:easy|0|0",
            "custom_tasks": "lighteval_custom/tasks/qa.py",
        }
    elif args.dataset == "ARC-C":
        task_kwargs = {
            "tasks": "custom|arc:challenge|0|0",
            "custom_tasks": "lighteval_custom/tasks/qa.py",
        }
    elif args.dataset == "HellaSwag":
        task_kwargs = {
            "tasks": "custom|hellaswag|0|0",
            "custom_tasks": "lighteval_custom/tasks/qa.py",
        }
    elif args.dataset == "LAMBADA":
        task_kwargs = {
            "tasks": "custom|lambada:openai|0|0",
            "custom_tasks": "lighteval_custom/tasks/qa.py",
        }
    elif args.dataset == "PIQA":
        task_kwargs = {
            "tasks": "custom|piqa|0|0",
            "custom_tasks": "lighteval_custom/tasks/qa.py",
        }
    # elif args.dataset == "WinoGrande":
    else:
        task_kwargs = {
            "tasks": "custom|winogrande|0|0",
            "custom_tasks": "lighteval_custom/tasks/qa.py",
        }
    

    results, details = vllm(
        model_config=model_config,
        use_chat_template=False,    # 这里修改为False，因为llama3没有chat_template模板
        # output_dir="./outputs/lighteval_outputs",
        max_samples=args.max_samples,
        load_responses_from_json_file=args.load_responses_from_json_file,
        **task_kwargs,
    )

    # save evaluation results
    eval_results = []
    task_name = list(details.keys())[0]
    for detail in details[task_name]:
        eval_results.append({
            "full_prompt": detail["full_prompt"],
            "generated_text": detail["predictions"][0],
            "gold": detail["gold"],
            "metrics": detail["metrics"]
        })
    if not args.debug and args.load_responses_from_json_file is None:
        with open(args.output_path, "w") as f:
            json.dump(eval_results, f, indent=4)
        print(f"Evaluation results saved at {args.output_path}.")


if __name__ == "__main__":
    args = parser_gen()
    main(args)
