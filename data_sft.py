# data_sft.py
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase


class JsonlSFTDataset(Dataset):
    """
    Expects JSONL with:
      {"prompt": "...", "output": "..."}
    Will build: f"{prompt}\n<assistant>\n{output}"
    """
    def __init__(self, path: str, tokenizer: PreTrainedTokenizerBase, max_len: int = 2048):
        self.items = []
        self.tok = tokenizer
        self.max_len = max_len
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj["prompt"].strip()
                output = obj["output"].strip()
                text = f"{prompt}\n<assistant>\n{output}"
                enc = self.tok(text, truncation=True, max_length=max_len, add_special_tokens=True)
                self.items.append(torch.tensor(enc["input_ids"], dtype=torch.long))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def pad_collate(batch, pad_id: int):
    maxlen = max(x.size(0) for x in batch)
    out = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), maxlen), dtype=torch.bool)
    for i, x in enumerate(batch):
        out[i, :x.size(0)] = x
        attn[i, :x.size(0)] = 1
    labels = out.clone()
    labels[~attn] = -100
    return out, attn, labels


def build_dataloader(path: str, tokenizer: PreTrainedTokenizerBase, max_len: int, batch_size: int, shuffle=True):
    ds = JsonlSFTDataset(path, tokenizer, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda b: pad_collate(b, tokenizer.pad_token_id))
