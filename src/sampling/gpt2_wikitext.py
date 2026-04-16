from __future__ import annotations

from collections.abc import Sequence

import torch
from modelscope.msdatasets import MsDataset
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _split_heads(x: torch.Tensor, n_head: int, head_dim: int) -> torch.Tensor:
    """[B, T, n_embd] -> [B, n_head, T, head_dim]"""
    b, t, _ = x.shape
    return x.view(b, t, n_head, head_dim).permute(0, 2, 1, 3).contiguous()


def collect_qk_gpt2_wikitext(
    *,
    model_path: str,
    n_docs: int = 256,
    max_length: int = 512,
    layers: Sequence[int] | None = None,
    heads: Sequence[int] | None = None,
    token_pos: int = -2,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    WikiText-2-raw-v1 文本过 frozen GPT-2，提取各层各 head 在 token_pos 处的 (q, k)。
    返回 [N, d_head], [N, d_head]。
    """
    ds = MsDataset.load("wikitext", subset_name="wikitext-2-v1", split="train")
    rows = [r["text"] for r in ds if r.get("text") and r["text"].strip()]
    tok = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval().to(device)

    n_head = model.config.n_head
    n_embd = model.config.n_embd
    head_dim = n_embd // n_head
    layer_list = list(range(model.config.n_layer)) if layers is None else list(layers)
    head_list = list(range(n_head)) if heads is None else list(heads)

    # 存储遍历时的当前 text 前向计算的q和k
    storage: dict[int, torch.Tensor] = {}

    def make_hook(li: int):
        def hook(_mod: nn.Module, _inp, out: torch.Tensor):
            q, k, _ = out.split(n_embd, dim=-1)
            storage[li] = torch.stack(
                [_split_heads(q, n_head, head_dim), _split_heads(k, n_head, head_dim)]
            ).detach()
        return hook

    hooks = []
    for li in layer_list:
        hooks.append(model.transformer.h[li].attn.c_attn.register_forward_hook(make_hook(li)))

    qs, ks = [], []
    dev = torch.device(device)

    with torch.no_grad():
        for text in tqdm(rows[:n_docs], desc="GPT-2 forward"):
            enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(dev)
            seq_len = input_ids.shape[1]
            # token_pos: -2 = 中点, -1 = 末尾, >=0 = 绝对位置
            pos = seq_len // 2 if token_pos == -2 else max(0, seq_len + token_pos) if token_pos < 0 else min(token_pos, seq_len - 1)

            storage.clear()
            model(input_ids, use_cache=False)

            for li in layer_list:
                if li not in storage:
                    continue
                both = storage[li]  # [2, B, H, T, Dh]
                for hi in head_list:
                    qs.append(both[0, 0, hi, pos, :].cpu())
                    ks.append(both[1, 0, hi, pos, :].cpu())

    for h in hooks:
        h.remove()

    return torch.stack(qs), torch.stack(ks)
    # 顺序: [
    #     sample1-layer1-head1, sample1-layer1-head2, ..., 
    #     sample1-layer2-head1, sample1-layer2-head2, ..., 
    #     sample2-layer1-head1, sample2-layer1-head2, ...,
    #     ...
    # ]
