"""
Test-wide setup: check tensor shapes and types at runtime.

The newer modules annotate their tensors with jaxtyping, for example
``Float[Tensor, "batch seq vocab"]``. While the tests run, jaxtyping's import hook wraps every
function in those modules with beartype, so a wrong shape, a wrong dtype or a mismatched
dimension name raises at the call instead of producing a silently wrong number later.

Set ``SHAPE_CHECKS=0`` to run the tests without it (for example to time them).
"""

from __future__ import annotations

import os

CHECKED_MODULES = [
    "src.inference",
    "src.models.lora",
    "src.models.modern",
    "src.optim",
    "src.post_training.dpo",
    "src.post_training.grpo",
]

if os.environ.get("SHAPE_CHECKS", "1") != "0":
    try:
        from jaxtyping import install_import_hook

        import tests._typecheck  # noqa: F401  (jaxtyping looks the checker up as tests -> _typecheck)
    except ImportError:  # beartype is a dev dependency; plain runs still work without it
        pass
    else:
        # Installed for the whole session (no `with`), before any test imports these modules.
        # tests/_typecheck.py is beartype with PEP 484 int-for-float promotion, like mypy.
        install_import_hook(CHECKED_MODULES, "tests._typecheck.typechecker")
