import torch
from transformers import BloomTokenizerFast, BloomForCausalLM
import torch.distributed as dist
import os
from utils import replace_8bit_linear_tp, get_8bit_tp_model
import time
import sys

# random_seed
torch.manual_seed(0)

def run_tp():
    # init
    world_size = 2
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = local_rank
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    model = BloomForCausalLM.from_pretrained("/data2/users/lczht/bloom-560m")
    model = model.half()
    # quantize
    model = replace_8bit_linear_tp(model).to(rank)

    # TP
    model = get_8bit_tp_model(model, rank, world_size)

    # inputs
    tokenizer = BloomTokenizerFast.from_pretrained("/data2/users/lczht/bloom-560m")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(rank)

    # inference
    outputs = model(**inputs, labels=inputs["input_ids"])
    logits = outputs.logits
    
    print(logits)


def compare():
     # init
    world_size = 2
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = local_rank
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    model = BloomForCausalLM.from_pretrained("/data2/users/lczht/bloom-560m")
    model = model.half()
    model = replace_8bit_linear_tp(model).to(rank)

    # TP
    model = get_8bit_tp_model(model, rank, world_size)

    # inputs
    tokenizer = BloomTokenizerFast.from_pretrained("/data2/users/lczht/bloom-560m")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(rank)

    # inference
    outputs = model(**inputs, labels=inputs["input_ids"])
    output = outputs.logits

    # reference model
    model2 = BloomForCausalLM.from_pretrained("/data2/users/lczht/bloom-560m", device_map='auto', load_in_8bit=True)
    outputs2 = model2(**inputs, labels=inputs["input_ids"])
    output2 = outputs2.logits

    assert torch.allclose(output, output2)==True, f'outputs from this method and hf method are not equal!'


def generate():
    # init
    world_size = 2
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = local_rank
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    num_tokens = 100
    model = BloomForCausalLM.from_pretrained("/data2/users/lczht/bloom-560m")
    model = model.half()
    # quantize
    model = replace_8bit_linear_tp(model)

    # TP
    model = get_8bit_tp_model(model, rank, world_size)
    model = model.to(rank)
    model2 = BloomForCausalLM.from_pretrained("/data2/users/lczht/bloom-560m", device_map='balanced_low_0', load_in_8bit=True)
    # inputs
    inputs = [
        "DeepSpeed is a machine learning framework",
        "He is working on",
        "He has a",
        "He got all",
        "Everyone is happy and I can",
        "The new movie that got Oscar this year",
        "In the far far distance from our galaxy,",
        "Peace is the only way",
        ]
    tokenizer = BloomTokenizerFast.from_pretrained("/data2/users/lczht/bloom-560m")
    input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(rank)


    generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False)
    
    sys.stdout = open(os.devnull, 'w')

    # inference
    t0 = time.time()
    outputs2 = model2.generate(**input_tokens, **generate_kwargs)
    t1 = time.time()
    outputs = model.generate(**input_tokens, **generate_kwargs)
    t2 = time.time()
    
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    outputs2 = tokenizer.batch_decode(outputs2, skip_special_tokens=True)
    
    sys.stdout = sys.__stdout__

    print(t1-t0, "ms")
    print(t2-t1, "ms")

if __name__ == '__main__':
    # run_tp()
    # compare()
    generate()
