"""
From-scratch post-training suite for ``train-llm-from-scratch``.

This package adds Supervised Fine-Tuning (SFT), reward modeling, PPO, DPO and
GRPO/RLVR on top of the repo's own custom :class:`~src.models.transformer.Transformer`.
Everything here is written in plain PyTorch in the same teaching style as the rest
of the repo -- no ``trl`` / ``peft`` / ``transformers`` for the algorithms.

Module map
----------
- ``chat_template``  : tiktoken r50k_base chat formatting + loss masking
- ``rollout``        : sampling + per-token log-prob utilities (the RL core)
- ``value_head``     : actor-critic wrapper for PPO
- ``reward_model``   : scalar reward head wrapper
- ``utils``          : frozen copies, checkpoint I/O, masked reductions, seeding
- ``distributed``    : optional pure-``torch.distributed`` DDP helpers
- ``sft`` / ``reward_train`` / ``dpo`` / ``ppo`` / ``grpo`` : the algorithms
- ``rewards``        : verifiable reward functions (GSM8K, arithmetic, format)
"""
