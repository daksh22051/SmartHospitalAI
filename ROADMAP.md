# Smart Hospital Orchestration - Project Roadmap

## Completed (Current Build)
- [x] OpenEnv-compatible environment API (`reset`, `step`, `state`)
- [x] Multi-task setup (`easy`, `medium`, `hard`)
- [x] Action/state/reward pipeline with dynamic events
- [x] Frontend-independent inference entrypoint
- [x] Programmatic grader with rubric scoring and pass/fail output
- [x] Optional LLM scoring hook with safe fallback mode
- [x] Timestamped evaluation history export
- [x] Docker training/evaluation service commands
- [x] True PyTorch PPO training command path (`main.py train --algorithm ppo`)
- [x] PPO checkpoint evaluation path in grader (direct model-policy evaluation)
- [x] Benchmark comparison generation support (`benchmark_comparison` in grader reports)
- [x] Benchmark dashboard generator (`evaluation/generate_benchmark_dashboard.py`)
- [x] Multi-agent extension baseline (`agent/multi_agent_extension.py`, `run_multi_agent_baseline.py`)
- [x] Hugging Face deployment automation script (`deploy_hf_space.ps1`)
- [x] Reward-ablation report generator (`validation/generate_reward_ablation_report.py`)
- [x] Stress-test scenario runner (`validation/run_stress_tests.py`)

## In Progress
- [ ] Final submission proof attachments (HF screenshots/log snippets)
- [ ] Security hygiene: rotate/revoke exposed API keys

## Next Iteration
- [ ] Multi-agent policy learning/training (beyond baseline coordinator)
- [ ] Auto-uploaded benchmark dashboard to Space UI
- [ ] Extended reward-ablation sweeps with plots and confidence intervals
- [ ] Long-horizon stress tests with container resource envelopes (2 CPU / 8 GB)
