"""The runtime type checker used by tests/conftest.py.

beartype with the PEP 484 numeric tower turned on, so an ``int`` is accepted where a ``float``
is annotated, exactly like mypy does. Without it, ``apply_lora(model, rank=4, alpha=8)``
would be rejected for passing 8 instead of 8.0.
"""

from beartype import BeartypeConf, beartype

typechecker = beartype(conf=BeartypeConf(is_pep484_tower=True))
