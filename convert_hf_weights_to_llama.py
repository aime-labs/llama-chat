# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
import argparse
import json
import os
import shutil

import torch


INTERMEDIATE_SIZE_MAP = {
    "7B": 11008,
    "13B": 13824,
    "30B": 17920,
    "65B": 22016,
}


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(model_path, input_base_path, model_size, num_gpus):
    assert model_size in INTERMEDIATE_SIZE_MAP
    os.makedirs(model_path, exist_ok=True)
    config = read_json(os.path.join(input_base_path, 'config.json'))
    dim = config['hidden_size']
    n_heads = config['num_attention_heads']
    n_layers = config['num_hidden_layers']
    dims_per_head = dim // n_heads

    def permute(w):
        return w.view(n_heads, 2, dim // n_heads // 2, dim).transpose(1, 2).reshape(dim, dim)

    state_dicts = [{} for i in range(num_gpus)]
    print('Converting weights.', end='', flush=True)
    
    for layer_i in range(n_layers):
        loaded = torch.load(os.path.join(input_base_path, f'pytorch_model-{layer_i+1:05d}-of-{n_layers + 1:05d}.bin'), map_location="cpu")
        print('.', end='', flush=True)
        for gpu_id, state_dict in enumerate(state_dicts):
            state_dict.update({
                f"layers.{layer_i}.attention_norm.weight": loaded[f"model.layers.{layer_i}.input_layernorm.weight"],
                f"layers.{layer_i}.ffn_norm.weight": loaded[f"model.layers.{layer_i}.post_attention_layernorm.weight"]
            })
            state_dict[f"layers.{layer_i}.attention.wq.weight"] = torch.chunk(permute(loaded[f"model.layers.{layer_i}.self_attn.q_proj.weight"]), num_gpus, dim=0)[gpu_id].clone()
            state_dict[f"layers.{layer_i}.attention.wk.weight"] = torch.chunk(permute(loaded[f"model.layers.{layer_i}.self_attn.k_proj.weight"]), num_gpus, dim=0)[gpu_id].clone()
            
            state_dict[f"layers.{layer_i}.attention.wv.weight"] = torch.chunk(loaded[f"model.layers.{layer_i}.self_attn.v_proj.weight"], num_gpus, dim=0)[gpu_id].clone()
            state_dict[f"layers.{layer_i}.attention.wo.weight"] = torch.chunk(loaded[f"model.layers.{layer_i}.self_attn.o_proj.weight"], num_gpus, dim=1)[gpu_id].clone()
            state_dict[f"layers.{layer_i}.feed_forward.w1.weight"] = torch.chunk(loaded[f"model.layers.{layer_i}.mlp.gate_proj.weight"], num_gpus, dim=0)[gpu_id].clone()
            state_dict[f"layers.{layer_i}.feed_forward.w2.weight"] = torch.chunk(loaded[f"model.layers.{layer_i}.mlp.down_proj.weight"], num_gpus, dim=1)[gpu_id].clone()
            state_dict[f"layers.{layer_i}.feed_forward.w3.weight"] = torch.chunk(loaded[f"model.layers.{layer_i}.mlp.up_proj.weight"], num_gpus, dim=0)[gpu_id].clone()


    loaded_last = torch.load(os.path.join(input_base_path, f'pytorch_model-{n_layers+1:05d}-of-{n_layers + 1:05d}.bin'), map_location="cpu")
    print('Done')
    print('Saving new checkpoint files..', end='', flush=True)
    for gpu_id, state_dict in enumerate(state_dicts):
        print('.', end='', flush=True)
        state_dict.update({
            "norm.weight": loaded_last["model.norm.weight"],
            "tok_embeddings.weight": torch.chunk(loaded_last["model.embed_tokens.weight"], num_gpus, dim=1)[gpu_id].clone(),
            "output.weight": torch.chunk(loaded_last["lm_head.weight"], num_gpus, dim=0)[gpu_id].clone()
            })
        torch.save(state_dict, os.path.join(model_path, f'merged.{num_gpus}GPUs.{gpu_id:02d}.pth'))
    print('Done')
    params = {
        'dim': dim,
        'multiple_of': 256,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'norm_eps': config['rms_norm_eps'],
        'vocab_size': -1
    }

    write_json(
        params,
        os.path.join(model_path, "params.json"),
    )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of LLama weights from huggingface",
    )
    parser.add_argument(
        "--model_size",
        choices=["7B", "13B", "30B", "65B"],
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write checkpoints and tokenizer",
    )
    parser.add_argument(
        "--num_gpus", type=int,
        help="Number of GPUs",
    )
    args = parser.parse_args()
    input_base_path = os.path.join(args.input_dir, "llama-{}-hf".format(args.model_size).lower())
    model_path = os.path.join(args.output_dir, args.model_size)
    write_model(
        model_path=model_path,
        input_base_path=input_base_path,
        model_size=args.model_size,
        num_gpus=args.num_gpus
    )
    shutil.copyfile(os.path.join(input_base_path, "tokenizer.model"), os.path.join(args.output_dir, "tokenizer.model"))


if __name__ == "__main__":
    main()