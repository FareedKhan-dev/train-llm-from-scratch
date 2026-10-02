"""DDP with Mixture of Experts: an expert that gets no tokens in a step must not crash training."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from src.models.modern import ModernConfig, ModernTransformer
from src.post_training.distributed import DDPContext, ddp_wrap


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
def test_ddp_wrap_lets_moe_experts_skip_a_step(tmp_path: Path) -> None:
    store = (tmp_path / "store").as_posix()
    dist.init_process_group("gloo", init_method=f"file:///{store.lstrip('/')}", rank=0, world_size=1)
    try:
        torch.manual_seed(0)
        cfg = ModernConfig(vocab_size=32, context_length=16, n_embed=32, n_head=4, n_blocks=1, n_experts=4, moe_top_k=1)
        model = ModernTransformer(cfg)
        with torch.no_grad():  # expert 0 wins every token, so experts 1 to 3 get no gradient
            router = model.blocks[0].mlp.router.weight
            router.zero_()
            router[0] = 1.0
        # world_size=2 in the context only so ddp_wrap wraps; the process group itself has one rank
        ddp = ddp_wrap(model, DDPContext(rank=0, local_rank=0, world_size=2, device="cpu"))
        idx = torch.randint(0, 32, (2, 8))
        for _ in range(2):  # without find_unused_parameters the second step raises
            _, loss = ddp(idx, idx)
            loss.backward()
    finally:
        dist.destroy_process_group()
