import json
import random

import torch
import torch_npu
import transformers
from datasets import load_dataset, Dataset


def read_jsonl(file_name):
    with open(file_name, mode='r') as reader:
        data = json.load(reader)
    return data


def get_pile_calib_dataset(tokenizer=None, n_samples=128, block_size=512):
    dataset = load_dataset("./datasets/pile-val-backup", split="validation")
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    split_samples = [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]
    data_dict = {
        "input_ids": [sample.tolist()[0] for sample in split_samples]
    }
    return Dataset.from_dict(data_dict)


def get_reasoning_calib_dataset(model_name=None, tokenizer=None, n_samples=128, seqlen=2048,
                                return_attention_mask=False):
    transformers.set_seed(seed=0)

    traindata = read_jsonl(f"./datasets/gen_data/{model_name}/NuminaMath-1.5.jsonl")
    traintext = ''
    for item in traindata:
        traintext += item['full_prompt'] + item['generated_text'][0] + "\n\n"

    trainenc = tokenizer(traintext, return_tensors='pt')    
    trainloader = []
    for _ in range(n_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)
    data_dict = {
        "input_ids": [sample.tolist()[0] for sample in trainloader]
    }
    if return_attention_mask:
        data_dict["attention_mask"] = [torch.ones_like(sample, dtype=torch.bool).tolist()[0] for sample in trainloader]
    return Dataset.from_dict(data_dict)
