---
title: SmartHospitalAI
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Smart Hospital Resource Orchestration  
*Advanced Reinforcement Learning Environment for Hospital Resource Management*

## 🏥 Problem Statement

Hospitals face critical resource allocation challenges daily. Emergency rooms overflow, ICU beds run scarce, and doctors become overwhelmed - leading to delayed treatments, increased mortality risks, and inefficient resource utilization. Traditional hospital management systems struggle with real-time optimization under uncertainty, resulting in suboptimal patient outcomes and wasted resources. This project addresses the fundamental challenge: **How can we optimize hospital resource allocation to save lives while maximizing efficiency?**

## 🎯 Solution Overview

Smart Hospital Resource Orchestration is a sophisticated reinforcement learning environment that simulates realistic hospital operations. Agents learn to make critical decisions about patient triage, resource allocation, and emergency response under pressure. The environment features dynamic patient arrivals, resource constraints, and multi-objective optimization that mirrors real hospital challenges. By training AI agents in this simulation, hospitals can develop intelligent decision support systems that improve patient outcomes, reduce wait times, and optimize resource utilization.

## 🎮 Action Space

The environment provides 5 discrete actions for hospital operations:

- **0 - WAIT**: No active intervention - useful when no immediate action needed
- **1 - ALLOCATE_RESOURCE**: Assign doctors and ICU beds to waiting patients based on priority
- **2 - ESCALATE_PRIORITY**: Increase severity of patients waiting too long (critical for safety)
- **3 - DEFER**: Defer normal-priority patients when system is overloaded (>80% capacity)
- **4 - REASSIGN**: Balance doctor workloads by moving patients between overloaded doctors

## 🏆 Reward System

Multi-objective reward system balancing patient outcomes and operational efficiency:

**Positive Rewards:**
- +10.0 for critical patients allocated within 2 steps (emergency response)
- +5.0 for emergency patients allocated successfully
- +2.0 for normal patients allocated
- +0.5 per patient with reduced wait time
- +0.3 for efficient resource utilization (70-95% capacity)

**Negative Penalties:**
- -5.0 for critical patients ignored while treating normal cases
- -2.0 per critical patient waiting >3 steps
- -1.0 for resource allocation conflicts
- -0.1 step penalty to encourage timely decisions

## 📊 Task Levels

Progressive difficulty scaling for curriculum learning:

### Easy Task
- ~5 initial patients (mostly normal, 1 emergency)
- 3 doctors, 5 beds (sufficient resources)
- No dynamic events (deterministic)
- 50 steps max - focus on basic operations

### Medium Task
- ~8-10 initial patients (mixed severity)
- 4 doctors, 6 beds (limited resources)
- Limited dynamic events enabled
- 75 steps max - introduces time pressure

### Hard Task
- ~10+ initial patients (critical-heavy)
- 5 doctors, 7 beds (severe constraints)
- Full dynamic events (emergencies, disruptions)
- 100 steps max - crisis management simulation

## 🚀 OpenEnv Compliance Quick Start

This project is compliant with the required OpenEnv interface:

- [`HospitalEnv.reset()`](smart_hospital_orchestration/environment/hospital_env.py:254)
- [`HospitalEnv.step()`](smart_hospital_orchestration/environment/hospital_env.py:351)
- [`HospitalEnv.state()`](smart_hospital_orchestration/environment/hospital_env.py:1195)

Importing `smart_hospital_orchestration` registers the Gymnasium task IDs automatically,
so the example below works as written after installation.

Run a full frontend-independent episode:

```bash
python inference.py --task medium --seed 42
```

The command prints a deterministic JSON summary line (`INFERENCE_RESULT=...`).

## 🚀 How to Run

### Quick Start (Local)
```bash
# Clone and setup
git clone <repository-url>
cd smart_hospital_orchestration
pip install -e .

# Optional: set API_BASE_URL to Groq-compatible OpenAI endpoint and provide GROQ_API_KEY
# API_BASE_URL=https://api.groq.com/openai/v1

# Run required OpenEnv inference entrypoint
python inference.py --task medium --seed 42

# Optional: baseline module
python -m smart_hospital_orchestration.inference.baseline_inference --task medium --episodes 1
```

### Docker Deployment
```bash
# Build container
docker build -t smart-hospital-orchestration .

# Run with default settings
docker run --rm smart-hospital-orchestration

# Custom parameters
docker run --rm smart-hospital-orchestration --task hard --seed 42
```

### Hugging Face Spaces Deployment
This project is ready for a Docker Space on Hugging Face.

1. Create a new Space and choose `Docker` as the SDK.
2. Push this repository with the `Dockerfile`, `openenv.yaml`, `README.md`, and source package intact.
3. Keep the default container command as `python inference.py --task medium --seed 42`.
4. Verify the Space builds successfully, then use the Space logs to confirm `INFERENCE_RESULT=...` is emitted.

### Advanced Usage
```python
import gymnasium as gym
import smart_hospital_orchestration

# Create environment
env = gym.make('SmartHospitalOrchestration-medium-v0')
state, info = env.reset()

# Run episode
for step in range(100):
    action = env.action_space.sample()  # Replace with your agent
  state, reward, done, info = env.step(action)

  if done:
        break
```

## 📈 Example Output

```
============================================================
RUNNING EPISODE: MEDIUM TASK
============================================================
Initial State: 9 patients, 4 doctors, 6 beds
------------------------------------------------------------
Step   1: ALLOCATE     | Reward:   8.30 | Patients:  7 | Waiting:  3 | Admitted:  4
Step   2: ALLOCATE     | Reward:   5.20 | Patients:  8 | Waiting:  2 | Admitted:  6
Step   3: ESCALATE     | Reward:  -0.50 | Patients:  9 | Waiting:  3 | Admitted:  6
Step   4: WAIT         | Reward:  -0.10 | Patients: 10 | Waiting:  4 | Admitted:  6

============================================================
BASELINE AGENT PERFORMANCE SUMMARY
============================================================
Total Reward: 42.15
Episode Length: 25 steps
Patients Treated: 12
  - Critical: 3
  - Emergency: 5
  - Normal: 4
Allocation Success Rate: 8/9 (88.9%)
Average Reward per Step: 1.686
============================================================
```

## 📁 Project Structure

```
smart_hospital_orchestration/
├── environment/           # Core simulation engine
│   ├── hospital_env.py    # Main environment class
│   ├── action_system.py   # Action handling logic
│   └── doctors.py         # Resource management
├── state/                 # State representation
│   └── state.py          # NumPy-based encoding
├── reward/               # Reward system
│   └── advanced_reward.py # Multi-objective rewards
├── tasks/                # Task configurations
│   └── advanced_tasks.py  # Progressive difficulty
├── inference/            # Baseline agents
│   └── baseline_inference.py # Rule-based agent
├── tests/               # Comprehensive test suite
├── openenv.yaml         # Environment metadata
├── Dockerfile           # Production deployment
└── requirements.txt     # Dependencies
```

## ⭐ Why This Project Stands Out

- **Real-World Impact**: Addresses actual hospital resource allocation challenges that affect patient survival
- **Sophisticated Simulation**: Dynamic patient arrivals, resource constraints, and realistic hospital operations
- **Multi-Objective Optimization**: Balances patient outcomes, efficiency, and resource utilization
- **Production Ready**: Complete Docker deployment, comprehensive testing, and professional documentation
- **Curriculum Learning**: Progressive difficulty from basic to crisis management scenarios
- **Advanced Reward Engineering**: 9-component reward system preventing gaming and encouraging optimal behavior

## 🔮 Optional Improvements

- **Multi-Agent Extension**: Coordinate multiple specialized agents (ER, ICU, surgery)
- **Deep RL Integration**: DQN, PPO, and A3C agent implementations
- **Real Hospital Data**: Integration with actual hospital operational data
- **Web Interface**: Interactive dashboard for visualization and control
- **Hospital Network**: Multi-hospital coordination and resource sharing

## 🧪 Testing

Run the comprehensive test suite:
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=smart_hospital_orchestration
```

## 📊 Performance Benchmarks

Baseline agent performance across difficulty levels:
- **Easy**: ~50-100 total reward (basic operations)
- **Medium**: ~0-50 total reward (moderate challenges)
- **Hard**: ~-50 to 0 total reward (crisis management)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Gymnasium](https://gymnasium.farama.org/) for RL environment standards
- Inspired by real hospital operations and medical triage principles
- Designed for advancing healthcare through intelligent resource management

---

**Built with ❤️ for improving healthcare through intelligent resource management**  

*This environment enables researchers and practitioners to develop and test AI systems that can save lives by optimizing hospital operations under pressure.* 🏥🤖
│   ├── state_normalizer.py
│   └── observation_space.py
├── reward/               # Reward computation
│   ├── reward_function.py
│   ├── reward_components.py
│   └── reward_shaping.py
├── tasks/                # Task configurations (easy, medium, hard)
│   ├── base_config.py
│   ├── easy_config.py
│   ├── medium_config.py
│   ├── hard_config.py
│   └── config_factory.py
├── agent/                # Agent policies and inference
│   ├── base_agent.py
│   ├── random_agent.py
│   ├── heuristic_agent.py
│   ├── policy_network.py
│   └── action_space.py
├── config/               # Configuration management
│   ├── config_loader.py
│   ├── config_validator.py
│   ├── default_config.py
│   └── configs/          # YAML configurations
│       ├── easy.yaml
│       ├── medium.yaml
│       └── hard.yaml
├── tests/                # Unit and integration tests
│   ├── test_environment.py
│   ├── test_state.py
│   ├── test_reward.py
│   ├── test_tasks.py
│   ├── test_agent.py
│   ├── test_config.py
│   └── test_integration.py
├── Dockerfile            # Container image definition
├── docker-compose.yml    # Multi-service orchestration
├── requirements.txt      # Production dependencies
├── requirements-dev.txt  # Development dependencies
├── pyproject.toml        # Package configuration
└── README.md             # This file
```

## Key Features

- **Modular Architecture**: Clean separation of concerns for easy extension and maintenance
- **OpenEnv Compatibility**: Designed for seamless integration with RL frameworks
- **Multiple Difficulty Levels**: Easy, medium, and hard task configurations
- **Comprehensive State Representation**: Rich observation space with normalization
- **Flexible Reward Design**: Modular reward components with shaping support
- **Agent Baselines**: Random and heuristic agents for comparison
- **YAML Configuration**: Human-readable configuration management
- **Docker Support**: Containerized deployment for reproducibility
- **Test Coverage**: Comprehensive unit and integration tests

## Installation

### From Source

```bash
git clone https://github.com/hospital-ai/smart-hospital-orchestration.git
cd smart-hospital-orchestration
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

### Docker

```bash
docker-compose up -d hospital-env
```

## Quick Start

### Basic Usage

```python
from smart_hospital_orchestration.environment import HospitalEnv

# Create environment with task name
env = HospitalEnv("easy")

# Run one episode with random actions
state, info = env.reset(seed=42)
done = False
total_reward = 0.0

while not done:
  action = env.action_space.sample()
  state, reward, done, info = env.step(action)
  total_reward += reward

print("Episode done. Total reward:", round(total_reward, 3))
env.close()
```

### Configuration

Load configuration from YAML:

```python
from smart_hospital_orchestration.config import ConfigLoader

loader = ConfigLoader()
config = loader.load("config/configs/easy.yaml")
```

Or use the Python API:

```python
from smart_hospital_orchestration.tasks import EasyTaskConfig

config = EasyTaskConfig().get_config()
```

## Task Configurations

### Easy
- 10 ICU beds, 5 doctors
- Low patient arrival rate (1.5/hour)
- Mostly low-priority patients
- Generous resource buffer

### Medium
- 20 ICU beds, 10 doctors
- Medium arrival rate (4/hour)
- Mixed priority distribution
- Tighter resource constraints

### Hard
- 30 ICU beds, 15 doctors
- High arrival rate (7/hour)
- High proportion of critical patients
- Equipment failures and emergencies
- Complex patient interactions

## Development

## Grader And LLM Scoring

Run deterministic grader scoring:

```bash
python -m smart_hospital_orchestration.evaluation.grader \
  --task medium --episodes 5 --seed 42 --policy heuristic \
  --rubric-profile hackathon_v1 --save-history \
  --output results/grader_result.json
```

Enable live LLM scoring (optional):

```bash
export LLM_GRADER_ENDPOINT="https://<your-endpoint>/v1/chat/completions"
export LLM_GRADER_API_KEY="<your-api-key>"
export LLM_GRADER_MODEL="gpt-4.1-mini"
```

Docker users:

1. Copy `.env.example` to `.env`.
2. Fill `LLM_GRADER_ENDPOINT` and `LLM_GRADER_API_KEY`.
3. Run the evaluation profile.

```bash
cp .env.example .env
docker compose --profile evaluation up --build evaluation
```

When `LLM_GRADER_ENDPOINT` is not set, the grader uses a safe local fallback score and still completes.

## Mandatory Environment Variables

Set these variables in your `.env` (or runtime environment):

- `API_BASE_URL` (OpenAI-compatible endpoint base, e.g. `https://<host>/v1`)
- `MODEL_NAME` (model name for baseline inference calls)
- `HF_TOKEN` (for Hugging Face deployment workflows)

Baseline and grader-specific optional vars:

- `LLM_GRADER_ENDPOINT`
- `LLM_GRADER_API_KEY`
- `LLM_GRADER_MODEL`

## Inference Script Output Contract

`inference.py` emits structured logs in this exact format:

- `[START] {...}`
- `[STEP] {...}`
- `[END] {...}`

It also prints a machine-readable summary line:

- `INFERENCE_RESULT={...}`

### PPO Training Tuning

Run true PyTorch PPO training with tunable hyperparameters:

```bash
python -m smart_hospital_orchestration.main train \
  --config config/configs/medium.yaml \
  --algorithm ppo \
  --timesteps 200000 \
  --save-path checkpoints/ppo_medium.pt \
  --ppo-lr 1e-4 \
  --ppo-gamma 0.995 \
  --ppo-gae-lambda 0.97 \
  --ppo-clip-eps 0.15 \
  --ppo-entropy-coef 0.002 \
  --ppo-value-coef 0.7 \
  --ppo-update-epochs 6 \
  --ppo-minibatch-size 128 \
  --ppo-rollout-steps 1024
```

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black smart_hospital_orchestration/
```

### Type Checking

```bash
mypy smart_hospital_orchestration/
```

## Architecture

The system follows a modular design with clear interfaces:

1. **Environment**: Core simulation logic implementing OpenEnv interface
2. **State**: Observation encoding, normalization, and space definition
3. **Reward**: Multi-component reward computation with shaping
4. **Tasks**: Pre-configured scenarios of varying difficulty
5. **Agent**: Policy interface with baseline implementations
6. **Config**: YAML-based configuration management

## OpenEnv Tasks

The three required tasks are implemented in [`HospitalEnv._load_task_config()`](smart_hospital_orchestration/environment/hospital_env.py:182):

- **Easy**: 50 steps, ~5 initial patients, 3 doctors, 5 beds, dynamic events disabled
- **Medium**: 75 steps, ~8–10 initial patients, 4 doctors, 6 beds, limited events
- **Hard**: 100 steps, ~10+ initial patients, 5 doctors, 7 beds, full events

Initial patient load is task-specific in [`HospitalEnv._generate_arrivals()`](smart_hospital_orchestration/environment/hospital_env.py:322).

## Contributing

Contributions are welcome! Please ensure:
- Code follows the existing style and patterns
- Tests are included for new functionality
- Documentation is updated as needed

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the Hospital AI Team.
