# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import time
import json
import argparse

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,

) -> LLaMA:
    start_time = time.time()
    #checkpoints = sorted(Path(ckpt_dir).glob(f'merged.{world_size}GPUs.*.pth'))
    checkpoints = sorted(Path(ckpt_dir).glob(f'consolidated.*.pth'))
    print('checkpoints', checkpoints)
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    if not tokenizer_path:
        tokenizer_path = Path(ckpt_dir).parent / 'tokenizer.model'
    tokenizer = Tokenizer(model_path=str(tokenizer_path))
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main():
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    args=load_flags()


    if not args.prompts:
        prompts = ["Give me a short answer to: I believe the meaning of life is" for _ in range(args.max_batch_size)]
        
    else:
        prompts = args.prompts.split(';')

    if not args.max_batch_size:
        args.max_batch_size = len(prompts)

    generator = load(
        args.ckpt_dir, args.tokenizer_path, local_rank, world_size, args.max_seq_len, args.max_batch_size
    )

    results, time_per_token = generator.generate(
        prompts, max_gen_len=512, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, repetition_penalty=args.repetition_penalty
    )

    #for result in results:
        #print(result)
        #print("\n==================================\n")
    print('bs: ', args.max_batch_size, 'time_per_token: ', time_per_token)

def load_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir", type=str, required=False,
        help="Location of LLama weights",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=False, default='/data/models/llama/tokenizer.model',
        help="Location of tokenizer"
    )
    parser.add_argument(
        '--temperature', type=float, default=0.8, required=False,
    help='Temperature'
                    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, required=False,
        help="Top_p, 0=<top_p<=1"
    )
    parser.add_argument(
        "--top_k", type=int, default=40, required=False,
        help="Top_k, 0=<top_k<=1",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=2048, required=False,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max_batch_size", type=int, required=False,
        help="Maximum batch size",
    )    
    parser.add_argument(
        "--repetition_penalty", type=float, default=(1.0/0.85), required=False,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--prompts", type=str, required=False,
        help="Prompt for text generation. Multiple prompts should be seperated by ;"
    )

    
    return parser.parse_args()


if __name__ == "__main__":
    main()