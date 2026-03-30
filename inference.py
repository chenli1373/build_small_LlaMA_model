from model.GPT_optimizer import Transformer, ModelArgs
import torch
import torch.nn as nn
import torch.optim as optim

from tokenizers import Tokenizer
import os
import re
from typing import Optional
from tqdm import tqdm

# from HF.GPTConfig import GPTConfig
# from HF.GPTModel import MyGPTModel

class GPT:
    def __init__(self, model: Transformer, tokenizer: Tokenizer, model_args: ModelArgs, optimizer, loss_fn):
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @staticmethod
    def build_model_train(checkpoints_dir: str, tokenizer_path: str, is_load_model: bool):
        tokenizer = Tokenizer.from_file(tokenizer_path)

        model_args: ModelArgs = ModelArgs(
            vocab_size=tokenizer.get_vocab_size(),
            split='inference'
        )

        model = Transformer(model_args).to(model_args.device)
        optimizer = optim.AdamW(model.parameters(), lr=model_args.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        if is_load_model:
            latest_ckpt, _ = GPT._get_latest_checkpoint(checkpoints_dir)
            if latest_ckpt is not None:
                print(f"Resuming training from {latest_ckpt}")
                checkpoint = torch.load(latest_ckpt, map_location=model_args.device)
                model.load_state_dict(checkpoint['model_state'], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                model_args.best_loss = checkpoint['val_loss']
                print(f"iter: {checkpoint['iter']}, train_loss: {checkpoint['train_loss']}, val_loss: {checkpoint['val_loss']}")

                # print("开始存储 HF 类模型")
                # config = GPTConfig(**vars(model_args))
                # hf_model = MyGPTModel(config)
                # hf_model.transformer.load_state_dict(checkpoint['model_state'], strict=False)
                # hf_save_dir = os.path.join(checkpoints_dir, 'hf_model')
                # hf_model.save_pretrained(hf_save_dir)
                # config.save_pretrained(hf_save_dir)
                # print("存储完成！")
        else:
            print("Training from scratch")
            if not os.path.exists(checkpoints_dir):
                print(f"Creating {checkpoints_dir}")
                os.makedirs("./optimed_GPT_ckpt", exist_ok=True)

        return GPT(model, tokenizer, model_args, optimizer, loss_fn)

    @staticmethod
    def _get_latest_checkpoint(ckpt_dir):
        if not os.path.exists(ckpt_dir):
            return None, 0
        ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("model_iter") and f.endswith(".pth")]
        if not ckpts:
            return None, 0

        iter_ckpt_pairs = []
        for f in ckpts:
            m = re.search(r'model_iter(\d+)\.pth', f)
            if m:
                iter_num = int(m.group(1))
                iter_ckpt_pairs.append((iter_num, f))

        if not iter_ckpt_pairs:
            return None, 0

        iter_ckpt_pairs.sort(key=lambda x: x[0], reverse=True)
        latest_iter, latest_file = iter_ckpt_pairs[0]
        return os.path.join(ckpt_dir, latest_file), latest_iter

    def inference(self, prompts: list[str], temperature=0.6, top_p=0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.model_args.max_seq_len - 1
        # 把每个 prompt 转换成 tokens
        prompt_tokens = [self.tokenizer.encode(prompt, add_special_tokens=True).ids for prompt in prompts]
        # prompt_tokens_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.model_args.device)
        batch_size = len(prompt_tokens)
        assert batch_size <= self.model_args.max_batch_size, f"batch size must be less than or equal to {self.model_args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.model_args.max_seq_len, f"prompt length must be less than or equal to {self.model_args.max_seq_len}"
        # 计算总生成长度
        total_len = min(self.model_args.max_seq_len, max_gen_len + max_prompt_len)

        # 创建一个 list，来放置要生成的 tokens 和 prompt tokens
        pad_id = self.tokenizer.token_to_id('<pad>')
        eos_id = self.tokenizer.token_to_id('<eos>')
        # 创建 (batch_size, total_len) 的 tensor，并用 <pad> 填充
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.model_args.device)
        # 先用 prompt_tokens 填充 tokens
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.model_args.device)

        # 表水是否到了任何 prompt 的句尾
        eos_reached = torch.tensor([False] * batch_size, device=self.model_args.device)
        prompt_tokens_mask = tokens != pad_id
        # 根据 prompt 逐个生成 token
        cur_iter = tqdm(range(1, total_len), desc="Generating tokens")
        for cur_pos in cur_iter:
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos - 1 : cur_pos], cur_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # 没有 temperature 就用贪婪策略
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # 只有当前 token 是 <pad> 时才替换
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # 预测下一个是 EOS
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == eos_id)
            if all(eos_reached):
                break

        # 解码输出
        out_tokens = []
        out_text = []
        for prompt_index, cur_prompt_tokens in enumerate(tokens.tolist()):
            # 在 EOS 处截断
            if eos_id in cur_prompt_tokens:
                eos_idx = cur_prompt_tokens.index(eos_id)
                cur_prompt_tokens = cur_prompt_tokens[: eos_idx]
            out_tokens.append(cur_prompt_tokens)
            out_text.append(self.tokenizer.decode(cur_prompt_tokens))
        return (out_tokens, out_text)

    def _sample_top_p(self, probs, p):
        # (batch_size, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # (batch_size, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # (batch_size, vocab_size)
        mask = probs_sum - probs_sort > p
        # 把所有没有被选到的 token 的概率都置零
        probs_sort[mask] = 0.0
        # 重新分配概率，使其相加为 1
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # 从 top p 个分布中采样一个
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # 得到 token 在词汇表中的位置
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token


if __name__ == '__main__':
    checkpoints_dir = "./optimed_GPT_ckpt"
    tokenizer_path = "./my_tokenizer.json"

    prompts = [
        "为什么要活着",
        "sakura会被怪兽打败吗",
        "愚者是否能回家"
    ]

    model = GPT.build_model_train(checkpoints_dir, tokenizer_path, True)

    out_tokens, out_texts = model.inference(prompts, max_gen_len=64)
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f"{out_texts[i]}")
        print('-' * 50)




