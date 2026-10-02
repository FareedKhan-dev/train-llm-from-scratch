"""
Every training script, end to end, exactly as a user runs them:

    pretrain -> SFT (+ LoRA) -> reward model -> DPO / SimPO -> PPO -> GRPO -> chat

Each script runs as a subprocess with the smoke configs (tiny model, CPU) on data generated
into a temp folder, for both architectures. This is the test that catches a broken flag, a
config field a trainer forgot, or a checkpoint one stage writes and the next cannot read.
Takes a couple of minutes on a laptop; skip it with `pytest -m "not slow"`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.prepare_rl_prompts import arithmetic_prompts
from src.models.modern import ModernTransformer
from src.models.transformer import Transformer
from src.post_training.chat_template import encode_chat
from src.post_training.inference import load_model_from_ckpt
from src.post_training.sft import pack_examples

pytestmark = pytest.mark.slow
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("e2e_data")
    rng = np.random.default_rng(0)
    for name in ("pile_train.h5", "pile_dev.h5"):
        with h5py.File(d / name, "w") as f:
            f.create_dataset("tokens", data=rng.integers(0, 50257, 40_000, dtype=np.int32))

    pairs = [(int(a), int(b)) for a, b in rng.integers(0, 20, (60, 2))]
    examples = [
        encode_chat([{"role": "user", "content": f"What is {a} + {b}?"},
                     {"role": "assistant", "content": f"<think>{a} + {b} = {a + b}</think><answer>{a + b}</answer>"}])
        for a, b in pairs
    ]
    for name, chunk in (("sft_packed.h5", examples), ("sft_dev_packed.h5", examples[:20])):
        tokens, masks = pack_examples(chunk * 4, 256)
        with h5py.File(d / name, "w") as f:
            f.create_dataset("tokens", data=tokens)
            f.create_dataset("loss_mask", data=masks)

    prefs = [{"prompt": f"What is {a} + {b}?", "chosen": f"<answer>{a + b}</answer>",
              "rejected": f"<answer>{a + b + 1}</answer>"} for a, b in pairs]
    rl = arithmetic_prompts(40, 9, seed=0)
    for name, rows in (("preferences.jsonl", prefs), ("preferences_test.jsonl", prefs[:16]),
                       ("rl_prompts_train.jsonl", rl), ("rl_prompts_test.jsonl", rl[:8]),
                       ("arithmetic_prompts.jsonl", rl)):
        (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return d


def run(script: str, *args: object) -> str:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, f"scripts/{script}", *map(str, args)], cwd=REPO, env=env,
                       capture_output=True, text=True, encoding="utf-8", timeout=900)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}"
    return r.stdout


@pytest.mark.parametrize("arch", ["classic", "modern"])
def test_every_stage_trains_and_hands_over_its_checkpoint(arch: str, data: Path, tmp_path: Path) -> None:
    ck = tmp_path / "models"
    common = ["--device", "cpu", "--log_dir", tmp_path / "logs", "--arch", arch]
    if arch == "modern":
        common += ["--n_kv_head", "2"]

    run("pretrain_base.py", "--config", "configs/smoke/pretrain.json", *common,
        "--train_path", data / "pile_train.h5", "--dev_path", data / "pile_dev.h5", "--out_ckpt", ck / "base.pt",
        "--train_steps", 6, "--eval_steps", 3, "--eval_iters", 2, "--save_every", 4,
        "--optimizer", "muon", "--lr_schedule", "wsd")
    sft = ["--config", "configs/smoke/sft.json", *common, "--pretrained_ckpt", ck / "base.pt",
           "--data_path", data / "sft_packed.h5", "--dev_path", data / "sft_dev_packed.h5",
           "--max_steps", 3, "--eval_steps", 2, "--grad_accum", 2]
    run("train_sft.py", *sft, "--out_ckpt", ck / "sft.pt")
    run("train_sft.py", *sft, "--out_ckpt", ck / "sft_lora.pt", "--lora_rank", 4)

    prefs = ["--sft_ckpt", ck / "sft.pt", "--pref_path", data / "preferences.jsonl",
             "--test_path", data / "preferences_test.jsonl", "--max_len", 128, "--eval_steps", 5]
    run("train_reward.py", "--config", "configs/smoke/reward.json", *common, *prefs, "--out_ckpt", ck / "reward.pt")
    run("train_dpo.py", "--config", "configs/smoke/dpo.json", *common, *prefs, "--out_ckpt", ck / "dpo.pt")
    run("train_dpo.py", "--config", "configs/smoke/dpo.json", *common, *prefs, "--out_ckpt", ck / "simpo.pt",
        "--loss_type", "simpo", "--beta", 2.0)

    rl = ["--sft_ckpt", ck / "sft.pt", "--prompt_path", data / "rl_prompts_train.jsonl",
          "--eval_prompt_path", data / "rl_prompts_test.jsonl", "--rollout_len", 16]
    run("train_ppo.py", "--config", "configs/smoke/ppo.json", *common, *rl, "--out_ckpt", ck / "ppo.pt",
        "--reward_source", "rm", "--reward_ckpt", ck / "reward.pt")
    run("train_grpo.py", "--config", "configs/smoke/grpo.json", *common, *rl, "--out_ckpt", ck / "grpo.pt",
        "--curriculum_path", data / "arithmetic_prompts.jsonl",
        "--adv_norm", "none", "--loss_agg", "seq-mean-token-sum-norm", "--clip_high", 0.28, "--filter_groups", "true")

    expected = ModernTransformer if arch == "modern" else Transformer
    for name in ("base", "sft", "sft_lora", "reward", "dpo", "simpo", "ppo", "grpo"):
        assert isinstance(load_model_from_ckpt(str(ck / f"{name}.pt"), "cpu"), expected), name

    reply = run("chat.py", "--ckpt", ck / "grpo.pt", "--prompt", "What is 2 + 3?", "--max_new_tokens", 8, "--device", "cpu")
    assert "loaded" in reply
