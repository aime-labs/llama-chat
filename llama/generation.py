# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer, chat_mode=False):
        self.model = model
        self.tokenizer = tokenizer
        self.chat_mode = chat_mode

    def generate(
            self,
            process_output_callback,
            prompts: List[str],
            max_gen_len: int,
            temperature: float = 0.8,
            top_p: float = 0.95,
            top_k: int = 40,
            repetition_penalty: float = (1.0 / 0.85),
            logprobs: bool = False,
            echo: bool = False,
    ) -> List[str]:
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)
        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )
        word = []
        generated_text = ""
        num_generated_tokens = 0
        for cur_pos in range(min_prompt_len, total_len):
            num_generated_tokens += 1
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_k(probs, top_p, top_k)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            if next_token != self.tokenizer.eos_id:
                word.append(next_token.item())
            word_str = self.tokenizer.decode(word)
            if " " in word_str:
                text = word_str[:word_str.find(" ") + 1]
                generated_text += text
                process_output_callback(generated_text, num_generated_tokens, False)
                word = word[-1:]


            tokens[:, cur_pos] = next_token
            mask = tokens != -1

            # replace -1 values with 2
            tokens = torch.where(mask, tokens, torch.ones_like(tokens)*self.tokenizer.eos_id)

            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )


            prev_pos = cur_pos
            if '\nUser:' in word_str:

                break
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            if all(eos_reached):
                break


            #print('ä', word_str)

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):

            start = 0 if echo else len(prompt_tokens[i])
            #toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            if self.tokenizer.pad_id in toks:
                pad_idx = toks.index(self.tokenizer.pad_id)
                toks = toks[:pad_idx]
                probs = probs[:pad_idx] if logprobs else None
            

            final_result = self.tokenizer.decode(toks).strip('User:')

            process_output_callback(final_result, num_generated_tokens, True)
            out_tokens.append(toks)
            output = ''
            
            out_logprobs.append(probs)
        
        return (out_tokens, out_logprobs if logprobs else None)



def sample_top_k(probs, top_p=0.0, top_k=40):
    if top_k > 0:
        probs_sort, probs_idx = torch.topk(probs, top_k)
    else:
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    if top_p > 0.0:
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)

    return next_token