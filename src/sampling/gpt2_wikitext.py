from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch import nn
from tqdm import tqdm


def _split_heads(x: torch.Tensor, n_head: int, head_dim: int) -> torch.Tensor:
    """[B, T, n_embd] -> [B, n_head, T, head_dim]"""
    b, t, _ = x.shape
    return x.view(b, t, n_head, head_dim).permute(0, 2, 1, 3).contiguous()


WIKITEXT_2_RAW_TRAIN_URLS = [
    # 首选：HF 官方 mirror（国内可达）
    "https://hf-mirror.com/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet",
    # 备用：HF 官方
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet",
]


def _load_wikitext_rows(n_docs: int) -> list[str]:
    """加载 wikitext-2-raw-v1 train split 的前 n_docs 条非空文本。

    依次尝试：
    1. ``datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")``
       （若 HF 可直连或已缓存；新版 ``datasets`` 禁用 script 需要 parquet 配置）
    2. 直接用 parquet URL 加载（经 hf-mirror）
    3. 本地 ``./wikitext-2-raw-v1/train.parquet``
    """
    from datasets import load_dataset  # lazy import
    from pathlib import Path

    last_err: Exception | None = None

    local_candidates = [
        Path("wikitext-2-raw-v1/train.parquet"),
        Path("wikitext-ms/wikitext-2-raw-v1/train-00000-of-00001.parquet"),
    ]
    for p in local_candidates:
        if p.exists():
            ds = load_dataset("parquet", data_files={"train": str(p)}, split="train")
            rows = [r["text"] for r in ds if r.get("text") and r["text"].strip()]
            return rows[:n_docs]

    for url in WIKITEXT_2_RAW_TRAIN_URLS:
        try:
            ds = load_dataset("parquet", data_files={"train": url}, split="train")
            rows = [r["text"] for r in ds if r.get("text") and r["text"].strip()]
            return rows[:n_docs]
        except Exception as e:
            last_err = e
            continue

    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        rows = [r["text"] for r in ds if r.get("text") and r["text"].strip()]
        return rows[:n_docs]
    except Exception as e:
        last_err = e

    raise RuntimeError(f"failed to load wikitext-2-raw-v1 train; last error: {last_err!r}")


def _load_gpt2(model_path: str):
    """Lazy import of transformers to keep Gaussian path independent."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy import
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tok, model


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
    rows = _load_wikitext_rows(n_docs)
    tok, model = _load_gpt2(model_path)
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


def iter_qkv_gpt2_wikitext(
    *,
    model_path: str,
    n_docs: int = 128,
    max_length: int = 512,
    layers: Sequence[int] | None = None,
    heads: Sequence[int] | None = None,
    device: str | torch.device = "cpu",
) -> Iterator[tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    流式产生每个 (doc, layer, head) 上完整序列的 (Q, K, V)。

    yield (doc_idx, layer, head, Q, K, V)，Q/K/V 形状 [1, T, d_head]。
    T 取决于该 doc 截断后的实际长度；调用方可直接喂 ``output_numer_denom``。
    """
    rows = _load_wikitext_rows(n_docs)
    tok, model = _load_gpt2(model_path)
    model.eval().to(device)

    n_head = model.config.n_head
    n_embd = model.config.n_embd
    head_dim = n_embd // n_head
    layer_list = list(range(model.config.n_layer)) if layers is None else list(layers)
    head_list = list(range(n_head)) if heads is None else list(heads)

    storage: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def make_hook(li: int):
        def hook(_mod: nn.Module, _inp, out: torch.Tensor):
            q, k, v = out.split(n_embd, dim=-1)
            storage[li] = (
                _split_heads(q, n_head, head_dim).detach(),
                _split_heads(k, n_head, head_dim).detach(),
                _split_heads(v, n_head, head_dim).detach(),
            )
        return hook

    hooks = [
        model.transformer.h[li].attn.c_attn.register_forward_hook(make_hook(li))
        for li in layer_list
    ]

    dev = torch.device(device)
    try:
        with torch.no_grad():
            for doc_idx, text in enumerate(tqdm(rows, desc="GPT-2 forward (seq)")):
                enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
                input_ids = enc["input_ids"].to(dev)
                storage.clear()
                model(input_ids, use_cache=False)
                for li in layer_list:
                    if li not in storage:
                        continue
                    q_all, k_all, v_all = storage[li]  # [1, H, T, Dh] each
                    for hi in head_list:
                        # 保留 batch=1 维：[1, T, Dh]
                        yield (
                            doc_idx,
                            li,
                            hi,
                            q_all[:, hi, :, :],
                            k_all[:, hi, :, :],
                            v_all[:, hi, :, :],
                        )
    finally:
        for h in hooks:
            h.remove()
