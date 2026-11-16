"""Benchmarks for base models: log-likelihood scoring as in lm-evaluation-harness."""

from src.benchmarks.scoring import Request, Score, encode_pair, score_requests
from src.benchmarks.tasks import TASKS, Example, Task, evaluate_task, load_examples

__all__ = [
    "TASKS",
    "Example",
    "Request",
    "Score",
    "Task",
    "encode_pair",
    "evaluate_task",
    "load_examples",
    "score_requests",
]
