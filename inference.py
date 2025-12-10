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
    parser.add_argument('--model', type=str, default='./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B',
                        help='Model to load.')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='dtype to use')
    # dataset
    parser.add_argument('--dataset', type=str, default='AIME-2024',
                        choices=["AIME-2024", "AIME-2025", "AIME-90", "MATH-500", "NuminaMath-1.5", "GSM8K", "GPQA-Diamond", "LiveCodeBench", "PPL"],
                        help='Dataset to load.')
    parser.add_argument('--max_samples', type=int, default=None, help='Max #samples (for debug)')
    # generation
    parser.add_argument('--temperature', type=float, default=0.6, help='Generation temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Generation top_p')
    parser.add_argument('--seed', type=int, default=42, help='Generation seed')
    parser.add_argument('--max_new_tokens', type=int, default=32768,
                        help='Maximum number of tokens to generate per output sequence.')
    parser.add_argument('--max_model_length', type=int, default=32768,
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
    

class PPLEvaluator:
    def __init__(self, args):
        self.args = args
        self.max_length = 2048
        # LLM head
        llm_hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=args.dtype)
        self.lm_head = llm_hf.lm_head
        # self.lm_head.to("cuda:0")
        self.lm_head.to("npu:0")
        # LLM to output hidden_states before LLM head
        self.llm = LLM(model=args.model, dtype=args.dtype, enforce_eager=True,
                       tensor_parallel_size=args.tensor_parallel_size, distributed_executor_backend="mp",
                       task="embed", override_pooler_config=PoolerConfig(pooling_type="ALL", normalize=False, softmax=False))

    @torch.no_grad()
    def __call__(self, testenc):
        print('Evaluating ppl...')
        testenc = testenc.input_ids
        nsamples = testenc.numel() // self.max_length

        nlls = []
        for i in tqdm(range(nsamples)):
            batch = {
                "prompt_token_ids": testenc[:, (i * self.max_length): ((i + 1) * self.max_length)].squeeze().tolist()
            }
            outputs = self.llm.encode(batch)
            hidden_states = outputs[0].outputs.data
            lm_logits = self.lm_head(hidden_states.to(self.lm_head.weight))
            shift_logits = lm_logits[:-1, :].contiguous()
            shift_labels = testenc[
                :, (i * self.max_length): ((i + 1) * self.max_length)
            ][:, 1:].to(shift_logits.device).squeeze()

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * self.max_length
            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * self.max_length))
        return ppl.item()


def main(args):
    if not args.debug and not args.overwrite and os.path.exists(args.output_path):
        print(f"Evaluation results found at {args.output_path}. Skip evaluation")
        return

    random.seed(args.seed)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    clean_up()
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
        seed=args.seed
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

    if args.dataset == "AIME-2024":
        task_kwargs = {
            "tasks": "custom|aime24|0|0",
            "custom_tasks": "lighteval_custom/tasks/reasoning.py",
        }
    elif args.dataset == "AIME-2025":
        task_kwargs = {
            "tasks": "custom|aime25|0|0",
            "custom_tasks": "lighteval_custom/tasks/reasoning.py",
        }
    elif args.dataset == "AIME-90":
        task_kwargs = {
            "tasks": "custom|aime90|0|0",
            "custom_tasks": "lighteval_custom/tasks/reasoning.py",
        }
    elif args.dataset == "MATH-500":
        task_kwargs = {
            "tasks": "custom|math_500|0|0",
            "custom_tasks": "lighteval_custom/tasks/reasoning.py",
        }
    elif args.dataset == "NuminaMath-1.5":
        task_kwargs = {
            "tasks": "custom|numina_math|0|0",
            "custom_tasks": "lighteval_custom/tasks/reasoning.py",
        }
    elif args.dataset == "GSM8K":
        task_kwargs = {
            "tasks": "custom|gsm8k|0|0",
            "custom_tasks": "lighteval_custom/tasks/reasoning.py",
        }
    elif args.dataset == "GPQA-Diamond":
        task_kwargs = {
            "tasks": "custom|gpqa:diamond|0|0",
            "custom_tasks": "lighteval_custom/tasks/reasoning.py",
        }
    elif args.dataset == "LiveCodeBench":
        task_kwargs = {
            "tasks": "custom|lcb:codegeneration|0|0",
            "custom_tasks": "lighteval_custom/tasks/livecodebench.py",
        }
    elif args.dataset == "PPL":
        from methods.utils import data_utils
        ppl_evaluator = PPLEvaluator(args)
        for eval_dataset in ["wikitext2"]:
        # for eval_dataset in ["wikitext2", "c4"]:
            print(eval_dataset)
            testloader = data_utils.get_loaders(
                eval_dataset,
                model=args.model,
                seqlen=2048,
                eval_mode=True
            )
            dataset_ppl = ppl_evaluator(testloader)
            print(dataset_ppl)
        return

    results, details = vllm(
        model_config=model_config,
        use_chat_template=True,
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
