"""PyTorch PPO trainer for Smart Hospital environment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "PyTorch is required for PPO training. Install with: pip install torch"
    ) from exc

from smart_hospital_orchestration.environment import HospitalEnv


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 64
    rollout_steps: int = 512
    max_grad_norm: float = 0.5


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, action_dim)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
    last_value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    adv = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(len(rewards))):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        adv[t] = gae
        next_value = values[t]
    returns = adv + values
    return adv, returns


def train_ppo(
    *,
    task: str,
    total_timesteps: int,
    seed: int = 42,
    save_path: str,
    config: PPOConfig | None = None,
) -> Dict[str, Any]:
    cfg = config or PPOConfig()
    env = HospitalEnv(task=task)
    state, _ = env.reset(seed=seed)

    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    obs = np.asarray(state["flat"], dtype=np.float32)
    step_count = 0
    episode_count = 0
    episode_rewards: List[float] = []
    current_episode_reward = 0.0

    while step_count < total_timesteps:
        rollout_obs: List[np.ndarray] = []
        rollout_actions: List[int] = []
        rollout_log_probs: List[float] = []
        rollout_rewards: List[float] = []
        rollout_dones: List[float] = []
        rollout_values: List[float] = []

        for _ in range(cfg.rollout_steps):
            if step_count >= total_timesteps:
                break

            obs_t = _to_tensor(obs, device).unsqueeze(0)
            with torch.no_grad():
                logits, value = model(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_state, reward, done, _ = env.step(int(action.item()))
            next_obs = np.asarray(next_state["flat"], dtype=np.float32)

            rollout_obs.append(obs.copy())
            rollout_actions.append(int(action.item()))
            rollout_log_probs.append(float(log_prob.item()))
            rollout_rewards.append(float(reward))
            rollout_dones.append(1.0 if done else 0.0)
            rollout_values.append(float(value.item()))

            obs = next_obs
            step_count += 1
            current_episode_reward += float(reward)

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                episode_count += 1
                reset_seed = seed + episode_count
                reset_state, _ = env.reset(seed=reset_seed)
                obs = np.asarray(reset_state["flat"], dtype=np.float32)

        with torch.no_grad():
            last_value = float(model(_to_tensor(obs, device).unsqueeze(0))[1].item())

        rewards_arr = np.asarray(rollout_rewards, dtype=np.float32)
        values_arr = np.asarray(rollout_values, dtype=np.float32)
        dones_arr = np.asarray(rollout_dones, dtype=np.float32)
        actions_arr = np.asarray(rollout_actions, dtype=np.int64)
        old_log_probs_arr = np.asarray(rollout_log_probs, dtype=np.float32)
        obs_arr = np.asarray(rollout_obs, dtype=np.float32)

        advantages, returns = _compute_gae(
            rewards=rewards_arr,
            values=values_arr,
            dones=dones_arr,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            last_value=last_value,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        idxs = np.arange(len(obs_arr))
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), cfg.minibatch_size):
                mb_idx = idxs[start:start + cfg.minibatch_size]
                mb_obs = _to_tensor(obs_arr[mb_idx], device)
                mb_actions = torch.as_tensor(actions_arr[mb_idx], dtype=torch.long, device=device)
                mb_old_log_probs = _to_tensor(old_log_probs_arr[mb_idx], device)
                mb_returns = _to_tensor(returns[mb_idx], device)
                mb_advantages = _to_tensor(advantages[mb_idx], device)

                logits, values = model(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, mb_returns)

                loss = policy_loss + (cfg.value_coef * value_loss) - (cfg.entropy_coef * entropy)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

    env.close()

    avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    checkpoint_path = Path(save_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "task": task,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "config": cfg.__dict__,
            "total_timesteps": step_count,
            "episodes_completed": episode_count,
            "avg_episode_reward": avg_reward,
        },
        checkpoint_path,
    )

    metadata_path = checkpoint_path.with_suffix(".json")
    metadata = {
        "artifact_type": "ppo_policy",
        "task": task,
        "checkpoint": str(checkpoint_path),
        "metadata_path": str(metadata_path),
        "total_timesteps": step_count,
        "episodes_completed": episode_count,
        "avg_episode_reward": round(avg_reward, 6),
        "policy": "ppo",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
