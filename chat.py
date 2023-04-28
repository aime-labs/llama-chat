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
    checkpoints = sorted(Path(ckpt_dir).glob(f'merged.{world_size}GPUs.*.pth'))
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
    tokenizer = Tokenizer(model_path=tokenizer_path)
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

    generator = load(
        args.ckpt_dir, args.tokenizer_path, local_rank, world_size, args.max_seq_len, args.max_batch_size
    )

    ctx = """A dialog, where User interacts with AI. AI is helpful, kind, obedient, honest and very reasonable.
User: Hello, AI.
AI: How can I assist you today?
"""        
    sampler = 'top_k' if not args.use_top_p else 'top_p'
    print(ctx)
    while True:
        prompts = []

        if local_rank == 0:
            ctx += "User: " + input(f'User: ') + "\n"
            prompts.append(ctx)
        else:
            prompts.append("")

        torch.distributed.broadcast_object_list(prompts, 0)

        results = generator.generate(
            prompts, max_gen_len=512, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, repetition_penalty=args.repetition_penalty, sampler=sampler
        )
        ctx = results[0]

def load_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir", type=str, required=False,
        help="Location of LLama weights",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=False,
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
        "--max_batch_size", type=int, default=1, required=False,
        help="Maximum batch size",
    )    
    parser.add_argument(
        "--repetition_penalty", type=float, default=(1.0/0.85), required=False,
        help="Top_k, 0=<top_k<=1",
    )
    parser.add_argument(
        '--use_top_p', action='store_true', required=False,
        help='Use top_p sampler instead of top_k'
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()