"""
Download + tokenize Pile-uncopyrighted shards into a single flat-token HDF5 for
pretraining. Faster than the original ``data_preprocess.py`` (which resizes the HDF5 per
document): here we stream-decompress, batch-tokenize with tiktoken, and write tokens to
the HDF5 in large chunks.

Writes to /ephemeral by default (the 1.5TB disk).

Examples:
    # dev split from the Pile validation file
    PYTHONPATH=. python scripts/prepare_pretrain_data.py --split val \
        --out /ephemeral/data/pile_dev.h5
    # one training shard
    PYTHONPATH=. python scripts/prepare_pretrain_data.py --split train --num_shards 1 \
        --out /ephemeral/data/pile_train.h5
"""

from __future__ import annotations

import argparse
import io
import json
import os

import h5py
import numpy as np
import requests
import tiktoken
import zstandard as zstd
from tqdm import tqdm

BASE_URL = "https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main"
EOT_ID = 50256
WRITE_CHUNK = 8_000_000  # flush tokens to HDF5 in ~8M-token chunks
ENC_BATCH = 1024          # documents per tiktoken batch-encode


def shard_urls(split: str, num_shards: int) -> list[str]:
    if split == "val":
        return [f"{BASE_URL}/val.jsonl.zst"]
    return [f"{BASE_URL}/train/{i:02d}.jsonl.zst" for i in range(num_shards)]


def download(url: str, dest: str) -> str:
    if os.path.exists(dest):
        print(f"  cached: {dest}")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  downloading {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest + ".part", "wb") as f:
            for chunk in tqdm(r.iter_content(1 << 20), total=total >> 20, unit="MB", desc="dl"):
                f.write(chunk)
    os.replace(dest + ".part", dest)
    return dest


def iter_texts(zst_path: str):
    dctx = zstd.ZstdDecompressor()
    with open(zst_path, "rb") as fh:
        reader = dctx.stream_reader(fh)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                txt = json.loads(line).get("text")
            except json.JSONDecodeError:
                continue
            if txt:
                yield txt


def tokenize_to_h5(zst_paths: list[str], out_path: str, max_tokens: int | None) -> int:
    enc = tiktoken.get_encoding("r50k_base")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    total = 0
    buf: list[int] = []
    with h5py.File(out_path, "w") as f:
        dset = f.create_dataset("tokens", (0,), maxshape=(None,), dtype="i4", chunks=(WRITE_CHUNK,))

        def flush():
            nonlocal total, buf
            if not buf:
                return
            arr = np.asarray(buf, dtype=np.int32)
            dset.resize(total + arr.size, axis=0)
            dset[total: total + arr.size] = arr
            total += arr.size
            buf = []

        for zp in zst_paths:
            print(f"  tokenizing {zp}")
            docs: list[str] = []
            pbar = tqdm(iter_texts(zp), unit="doc", desc="tok")
            for txt in pbar:
                docs.append(txt)
                if len(docs) >= ENC_BATCH:
                    for ids in enc.encode_ordinary_batch(docs):
                        buf.extend(ids)
                        buf.append(EOT_ID)
                    docs = []
                    if len(buf) >= WRITE_CHUNK:
                        flush()
                        pbar.set_postfix(tokens=f"{total/1e6:.1f}M")
                    if max_tokens and total >= max_tokens:
                        break
            if docs:
                for ids in enc.encode_ordinary_batch(docs):
                    buf.extend(ids)
                    buf.append(EOT_ID)
            flush()
            if max_tokens and total >= max_tokens:
                break
    print(f"  wrote {total:,} tokens -> {out_path}")
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--num_shards", type=int, default=1, help="train shards to use")
    p.add_argument("--raw_dir", default="/ephemeral/data/pile_raw")
    p.add_argument("--out", required=True)
    p.add_argument("--max_tokens", type=int, default=None, help="stop after this many tokens")
    args = p.parse_args()

    urls = shard_urls(args.split, args.num_shards)
    local = []
    for u in urls:
        name = "val.jsonl.zst" if args.split == "val" else os.path.basename(u)
        local.append(download(u, os.path.join(args.raw_dir, name)))
    tokenize_to_h5(local, args.out, args.max_tokens)


if __name__ == "__main__":
    main()
