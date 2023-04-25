
import torch
from pathlib import Path
from deepdiff import DeepDiff
import json
import os

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def compare_models(checkpoint1, checkpoint2):
    models_differ = 0
    k=1
    for key_item_1, key_item_2 in zip(checkpoint1.items(), checkpoint2.items()):

        if key_item_1[1].dtype == key_item_2[1].dtype:
            pass
        else:
            print('precision difference')
        if key_item_1[1].shape == key_item_2[1].shape:
            pass
        else:
            print('shape difference')

        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                print('Mismtach key found at', key_item_1[0], key_item_2[0])
    if models_differ == 0:
        print('Models match perfectly! :)')

params = read_json(os.path.join('/data/models/llama/65B', "params.json"))

n_layers = params["n_layers"]
ckpt_dir_hf='/data/models/llama/test/65B/'
ckpt_dir_llama='/data/models/llama/65B/'
checkpoints_hf = sorted(Path(ckpt_dir_hf).glob(f'merged.4GPUs.*.pth'))
checkpoints_llama = sorted(Path(ckpt_dir_llama).glob(f'merged.4GPUs.*.pth'))
print('checkpoints_hf', checkpoints_hf)
print('checkpoints_llama', checkpoints_llama)

#DeepDiff(checkpoint_hf.items(), checkpoint_llama.items())
#for i in range(4):
ckpt_path_hf = checkpoints_hf[2]
ckpt_path_llama = checkpoints_llama[2]
print('####', ckpt_path_hf==ckpt_path_llama)
checkpoint_hf = torch.load(ckpt_path_hf, map_location="cpu")
checkpoint_llama = torch.load(ckpt_path_llama, map_location="cpu")

compare_models(checkpoint_hf, checkpoint_llama)
#for i, j in zip(checkpoint_hf, checkpoint_llama):
#    print(i,j)