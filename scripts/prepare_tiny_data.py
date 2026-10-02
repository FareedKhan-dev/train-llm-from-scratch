"""
Prepare a small dataset for training on a laptop CPU (the student track).

Datasets (--dataset):
    tinystories   short, simple stories made for training tiny models (Eldan and Li, 2023).
                  Only the first --max_mb megabytes of the training file are downloaded.
    shakespeare   the 1 MB Tiny Shakespeare file.
    text          any UTF-8 text file of your own (--input path/to/file.txt).

Tokenizers (--tokenizer):
    bpe           (default) train a byte-level BPE on the training text, from scratch, with
                  --vocab_size ids (4096 by default). A small vocabulary keeps the embedding
                  and output layers small, which is what makes tiny models fast on a CPU.
    r50k_base     the GPT-2/3 tokenizer used by the rest of the repo (also cl100k_base, o200k_base).

Output (same format as the Pile files, so every loader in the repo reads it):
    data/tiny/train.h5      flat int32 token stream, documents separated by <|endoftext|>
    data/tiny/val.h5        held-out tokens for the dev loss
    data/tiny/tokenizer.json  (BPE only)
The tokenizer description is also stored in the HDF5 attributes, so the trainer and the
generation script always use the same tokenizer as the data.

Examples:
    python scripts/prepare_tiny_data.py                                  # TinyStories + BPE 4096
    python scripts/prepare_tiny_data.py --dataset shakespeare --vocab_size 2048
    python scripts/prepare_tiny_data.py --dataset text --input my_notes.txt --tokenizer r50k_base
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import requests
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.tokenizer import EOT, BPETokenizer, tokenizer_from_spec, tokenizer_spec  # noqa: E402

TINYSTORIES = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{split}.txt"
SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download(url: str, dest: Path, max_bytes: int | None = None) -> Path:
    """Stream ``url`` to ``dest`` (resumable cache: an existing file is reused)."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}" + (f" (first {max_bytes / 1e6:.0f} MB)" if max_bytes else ""))
    part = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        if max_bytes and total:
            total = min(total, max_bytes)
        written = 0
        with open(part, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="  download") as bar:
            for chunk in r.iter_content(1 << 20):
                if max_bytes and written + len(chunk) > max_bytes:
                    chunk = chunk[: max_bytes - written]
                f.write(chunk)
                written += len(chunk)
                bar.update(len(chunk))
                if max_bytes and written >= max_bytes:
                    break
    os.replace(part, dest)
    return dest


def split_stories(raw: str) -> list[str]:
    """TinyStories separates stories with <|endoftext|> lines. Drop a cut-off last story."""
    stories = [s.strip() for s in raw.split(EOT)]
    if not raw.rstrip().endswith(EOT) and len(stories) > 1:
        stories = stories[:-1]  # the download stopped in the middle of this one
    return [s for s in stories if s]


def split_text(raw: str, val_frac: float) -> tuple[list[str], list[str]]:
    """Split one long text into train / val at a line break near ``1 - val_frac``."""
    cut = raw.rfind("\n", 0, int(len(raw) * (1 - val_frac)))
    cut = cut if cut > 0 else int(len(raw) * (1 - val_frac))
    return [raw[:cut]], [raw[cut:]]


def load_documents(args: argparse.Namespace) -> tuple[list[str], list[str], str]:
    raw_dir = Path(args.raw_dir)
    if args.dataset == "tinystories":
        max_bytes = int(args.max_mb * 1e6)
        train_file = download(TINYSTORIES.format(split="train"), raw_dir / f"tinystories_train_{args.max_mb:g}mb.txt", max_bytes)
        val_file = download(TINYSTORIES.format(split="valid"), raw_dir / "tinystories_valid.txt")
        train = split_stories(train_file.read_text(encoding="utf-8", errors="replace"))
        val = split_stories(val_file.read_text(encoding="utf-8", errors="replace"))[: args.max_val_docs]
        return train, val, "TinyStories V2 (roneneldan/TinyStories)"
    if args.dataset == "shakespeare":
        path = download(SHAKESPEARE, raw_dir / "tinyshakespeare.txt")
        train, val = split_text(path.read_text(encoding="utf-8"), args.val_frac)
        return train, val, "Tiny Shakespeare"
    if not args.input:
        raise SystemExit("--dataset text needs --input path/to/your.txt")
    train, val = split_text(Path(args.input).read_text(encoding="utf-8"), args.val_frac)
    return train, val, f"text file {args.input}"


def build_tokenizer(args: argparse.Namespace, train_docs: list[str]):
    if args.tokenizer != "bpe":
        return tokenizer_from_spec({"type": "tiktoken", "name": args.tokenizer})
    # Train on (at most) the first --bpe_train_mb megabytes; more text barely changes the merges.
    budget, sample = int(args.bpe_train_mb * 1e6), []
    for doc in train_docs:
        sample.append(doc)
        budget -= len(doc)
        if budget <= 0:
            break
    print(f"Training a byte-level BPE tokenizer: vocab {args.vocab_size} on {sum(map(len, sample)) / 1e6:.1f} MB ...")
    t0 = time.perf_counter()
    tok = BPETokenizer.train(sample, args.vocab_size, verbose=True)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    return tok


def encode_documents(tok, docs: list[str], desc: str) -> np.ndarray:
    eot = tok.eot_token
    ids: list[int] = []
    for doc in tqdm(docs, desc=f"  encode {desc}", unit="doc", mininterval=1.0):
        ids.extend(tok.encode_ordinary(doc))
        ids.append(eot)
    return np.asarray(ids, dtype=np.int32)


def write_h5(path: Path, tokens: np.ndarray, spec: dict, source: str, n_docs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        dset = f.create_dataset("tokens", data=tokens)
        dset.attrs["tokenizer"] = json.dumps(spec)
        dset.attrs["vocab_size"] = int(tokenizer_from_spec(spec).n_vocab)
        dset.attrs["source"] = source
        dset.attrs["n_documents"] = n_docs
    print(f"  wrote {tokens.size:,} tokens -> {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["tinystories", "shakespeare", "text"], default="tinystories")
    p.add_argument("--input", default=None, help="your text file (with --dataset text)")
    p.add_argument("--out_dir", default="data/tiny")
    p.add_argument("--raw_dir", default="data/raw", help="where downloads are cached")
    p.add_argument("--max_mb", type=float, default=25, help="TinyStories: MB of training text to download")
    p.add_argument("--max_val_docs", type=int, default=5000, help="TinyStories: stories kept for validation")
    p.add_argument("--val_frac", type=float, default=0.1, help="shakespeare / text: fraction held out")
    p.add_argument("--tokenizer", default="bpe", help="bpe, r50k_base, cl100k_base or o200k_base")
    p.add_argument("--vocab_size", type=int, default=4096, help="BPE vocabulary size")
    p.add_argument("--bpe_train_mb", type=float, default=10, help="MB of text used to train the BPE")
    args = p.parse_args()

    train_docs, val_docs, source = load_documents(args)
    print(f"{source}: {len(train_docs):,} train / {len(val_docs):,} val documents")

    tok = build_tokenizer(args, train_docs)
    spec = tokenizer_spec(tok)
    out = Path(args.out_dir)
    if isinstance(tok, BPETokenizer):
        tok.save(out / "tokenizer.json")
        print(f"  saved tokenizer -> {out / 'tokenizer.json'}")

    train_ids = encode_documents(tok, train_docs, "train")
    val_ids = encode_documents(tok, val_docs, "val")
    write_h5(out / "train.h5", train_ids, spec, source, len(train_docs))
    write_h5(out / "val.h5", val_ids, spec, source, len(val_docs))

    n_chars = sum(map(len, train_docs))
    print(f"Done. {n_chars / max(1, train_ids.size):.2f} characters per token, vocab {tok.n_vocab}.")
    print("Next: python scripts/train_transformer.py --preset student")


if __name__ == "__main__":
    main()
