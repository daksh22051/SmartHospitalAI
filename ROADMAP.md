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

## In Progress
- [ ] Strengthen PPO checkpoint evaluation path inside grader (direct model-policy evaluation mode)
- [ ] Add benchmark comparison dashboard across random/heuristic/ppo policies

## Next Iteration
- [ ] Multi-agent extension (ER + ICU coordination)
- [ ] Hugging Face Space deployment automation
- [ ] Extended reward-ablation experiments and reports
- [ ] Additional stress-test scenarios for large hospital loads
