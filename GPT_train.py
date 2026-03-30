from model.GPT_optimizer import Transformer, ModelArgs
import torch
import torch.nn as nn
import torch.optim as optim

from tokenizers import Tokenizer
import os
from torch.utils.tensorboard import SummaryWriter
import re

def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    input = torch.stack([data[i : i + block_size] for i in ix])
    label = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    input, label = input.to(device), label.to(device)
    return input, label

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

        model_args: ModelArgs = ModelArgs(vocab_size=tokenizer.get_vocab_size())

        model = Transformer(model_args).to(model_args.device)
        optimizer = optim.AdamW(model.parameters(), lr=model_args.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        start_iter = 0

        if is_load_model:
            latest_ckpt, start_iter = GPT._get_latest_checkpoint(checkpoints_dir)
            if latest_ckpt is not None:
                print(f"Resuming training from {latest_ckpt} (iter {start_iter})")
                checkpoint = torch.load(latest_ckpt, map_location=model_args.device)
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                model_args.best_loss = checkpoint['val_loss']
        else:
            print("Training from scratch")
            if not os.path.exists(checkpoints_dir):
                print(f"Creating {checkpoints_dir}")
                os.makedirs("./optimed_GPT_ckpt", exist_ok=True)

        return GPT(model, tokenizer, model_args, optimizer, loss_fn), start_iter

    def train(self, args: ModelArgs, train_loader: torch.Tensor, val_loader: torch.Tensor, writer: SummaryWriter, tokenizer_path: str, checkpoints_dir: str, start_iter):
        self.model.train()
        # input, label: (batch_size, seq_len)
        if start_iter > args.max_iters:
            print(f"start_iter: {start_iter} > max_iters: {args.max_iters}")
            exit()
        for iter in range(start_iter, args.max_iters):

            # 周期性评估
            if (iter % args.eval_interval == 0 or iter == args.max_iters - 1) and iter > start_iter:
                losses = self.evaluate_loss(train_loader, val_loader, args)

                writer.add_scalar('eval/train_loss', losses['train'], iter)
                writer.add_scalar('eval/val_loss', losses['val'], iter)

                print(f"step {iter} | train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

                if losses['val'] < args.best_loss:
                    print('Saving model...')
                    args.best_loss = losses['val']
                    self.save_model(checkpoints_dir, iter, losses['train'], losses['val'], tokenizer_path)

            # 得到 batch_size 的数据
            input, label = get_batch(train_data, args.batch_size, args.block_size, args.device)
            # (batch_size, seq_len, vocab_size)
            logits = self.model(input, start_pos=0)

            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(batch_size * seq_len, vocab_size)
            labels = label.view(batch_size * seq_len)
            loss = self.loss_fn(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if iter < 100:
                print(f"step {iter} | train loss: {loss:.4f}")

            writer.add_scalar("train/loss", loss.item(), iter)

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

    @torch.no_grad()
    def evaluate_loss(self, train_loader, val_loader, args: ModelArgs):
        self.model.eval()

        out = {}
        for split, loader in [('train', train_loader), ('val', val_loader)]:
            losses = torch.zeros(args.eval_iters)
            for iter in range(args.eval_iters):
                input, label = get_batch(data, args.batch_size, args.block_size, args.device)

                logits = self.model(input, start_pos=0)
                batch_size, seq_len, vocab_size = logits.shape
                logits = logits.view(batch_size * seq_len, vocab_size)
                labels = label.view(batch_size * seq_len)

                loss = self.loss_fn(logits, labels)
                losses[iter] = loss.item()

            out[split] = losses.mean()

        self.model.train()
        torch.cuda.empty_cache()
        return out

    def save_model(self, path: str, iter: int, train_loss: float, val_loss: float, tokenizer_path: str):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f"model_iter{iter}.pth")
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'iter': iter,
            'model_args': vars(self.model_args),
            'tokenizer_path': tokenizer_path,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        torch.save(checkpoint, path)

if __name__ == '__main__':
    checkpoints_dir = "./optimed_GPT_ckpt"
    tokenizer_path = "./my_tokenizer.json"

    model, start_iter = GPT.build_model_train(checkpoints_dir, tokenizer_path, True)

    with open('optimed_GPT_input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    ids = model.tokenizer.encode(text).ids
    data = torch.tensor(ids, dtype=torch.long)

    # 90% 为训练集，10% 为验证集
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    writer = SummaryWriter(log_dir="./runs/optimed_gpt_decoder")

    model.train(model.model_args, train_data, val_data, writer, tokenizer_path, checkpoints_dir, start_iter)

    writer.close()



