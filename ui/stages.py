"""
Single source of truth for the control panel: maps each pipeline stage to its config
dataclass, JSON file, training script, log prefix, theory doc, diagram, and whether it is a
multi-GPU (torchrun) stage. Every page / form / job reads from here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from config.post_training_config import (
    PretrainConfig, SFTConfig, RewardConfig, DPOConfig, PPOConfig, GRPOConfig,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = "/ephemeral/ckpts"
DATA_DIR = "/ephemeral/data"
LOG_DIR = "/ephemeral/logs"


@dataclass(frozen=True)
class Stage:
    key: str                 # short id ("sft")
    title: str               # display ("Supervised Fine-Tuning")
    emoji: str
    cfg_cls: type            # the dataclass
    config_json: str         # configs/<stage>.json
    smoke_json: str          # configs/smoke/<stage>.json
    script: str              # scripts/<trainer>.py
    log_prefix: str          # JSONL files are <log_prefix>_<ts>.jsonl
    doc_md: str              # docs/<NN_stage>.md (theory)
    diagram_png: str         # docs/diagrams/<NN_stage>.png
    multi_gpu: bool          # True -> can use torchrun


STAGES: dict[str, Stage] = {
    "pretrain": Stage("pretrain", "Pretraining", "📚", PretrainConfig,
                      "configs/pretrain.json", "configs/smoke/pretrain.json",
                      "scripts/pretrain_base.py", "pretrain",
                      "docs/02_pretraining.md", "docs/diagrams/02_pretraining.png", True),
    "sft": Stage("sft", "Supervised Fine-Tuning", "🎯", SFTConfig,
                 "configs/sft.json", "configs/smoke/sft.json",
                 "scripts/train_sft.py", "sft",
                 "docs/03_sft.md", "docs/diagrams/03_sft.png", True),
    "reward": Stage("reward", "Reward Model", "🏅", RewardConfig,
                    "configs/reward.json", "configs/smoke/reward.json",
                    "scripts/train_reward.py", "reward",
                    "docs/04_reward_model.md", "docs/diagrams/04_reward_model.png", True),
    "dpo": Stage("dpo", "DPO / ORPO / KTO", "⚖️", DPOConfig,
                 "configs/dpo.json", "configs/smoke/dpo.json",
                 "scripts/train_dpo.py", "dpo",
                 "docs/05_dpo.md", "docs/diagrams/05_dpo.png", True),
    "ppo": Stage("ppo", "PPO (RLHF)", "🎮", PPOConfig,
                 "configs/ppo.json", "configs/smoke/ppo.json",
                 "scripts/train_ppo.py", "ppo",
                 "docs/06_ppo.md", "docs/diagrams/06_ppo.png", True),
    "grpo": Stage("grpo", "GRPO / RLVR", "🧠", GRPOConfig,
                  "configs/grpo.json", "configs/smoke/grpo.json",
                  "scripts/train_grpo.py", "grpo",
                  "docs/07_grpo.md", "docs/diagrams/07_grpo.png", True),
}

# Data-prep scripts (plain python jobs, no config dataclass).
DATA_SCRIPTS = {
    "Pretrain corpus (Pile → HDF5)": ["scripts/prepare_pretrain_data.py", "--split", "train",
                                      "--num_shards", "1", "--out", f"{DATA_DIR}/pile_train.h5"],
    "Pretrain dev (Pile val)": ["scripts/prepare_pretrain_data.py", "--split", "val",
                                "--out", f"{DATA_DIR}/pile_dev.h5"],
    "SFT (Alpaca · Dolly · GSM8K)": ["scripts/prepare_sft_data.py"],
    "Preferences (HH-RLHF · UltraFeedback)": ["scripts/prepare_preference_data.py", "--source", "both"],
    "RL prompts (GSM8K + arithmetic)": ["scripts/prepare_rl_prompts.py"],
}

ABS_DOC = lambda rel: os.path.join(REPO_ROOT, rel)   # noqa: E731
