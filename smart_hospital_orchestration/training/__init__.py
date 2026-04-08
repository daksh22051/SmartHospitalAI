"""Training utilities for Smart Hospital Orchestration."""

def train_ppo(*args, **kwargs):
	from .ppo_trainer import train_ppo as _train_ppo

	return _train_ppo(*args, **kwargs)

__all__ = ["train_ppo"]
