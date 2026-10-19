"""Evaluation harness: datasets, per-transform scoring, method x task sweeps."""
from .datasets import PairFolder
from .harness import SCHEMA, evaluate, run_matrix

__all__ = ["PairFolder", "evaluate", "run_matrix", "SCHEMA"]
