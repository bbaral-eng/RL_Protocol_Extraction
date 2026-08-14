# Recurrent Reinforcement Learning for Digital Microfluidic Protocol Extraction

This repository implements recurrent reinforcement learning algorithms for protocol reverse-engineering on Digital Microfluidic Biochips (DMFBs). Agents learn to infer hidden droplet protocols by strategically probing electromagnetic field (EMF) measurements in a partially observable environment.

## Problem Statement

Agents operate in a Partially Observable Markov Decision Process (POMDP) with:
- **Spatial Ambiguity**: Single EMF readings correspond to multiple possible droplet positions
- **Temporal Dependencies**: Droplet motion has velocity constraints and physics-based dynamics
- **Hidden Protocol**: The true droplet path is unknown and must be reverse-engineered

## Implemented Algorithms

### 1. Recurrent Deep Q-Network (DRQN)
- **Architecture**: LSTM-based recurrent memory (256 hidden units, 1 layer)
- **Features**: Double DQN, Dueling Network, Prioritized Experience Replay
- **Action Masking**: Belief-guided masking via Q-value manipulation
- **File**: `RL_Protocol_Extraction_Agents/RecurrentDQN.ipynb`

### 2. Recurrent Proximal Policy Optimization (PPO)
- **Architecture**: LSTM-based recurrent memory (256 hidden units, 1 layer)
- **Features**: On-policy learning with GAE, separate actor-critic heads
- **Action Masking**: Belief-guided masking via logit manipulation
- **File**: `RL_Protocol_Extraction_Agents/RecurrentPPO.ipynb`

### 3. Baseline Algorithms
- **Methods**: Random, Greedy, Entropy-Weighted, Standard DQN, Standard PPO
- **File**: `benchmarking/baselines.ipynb`

## Prerequisites

- Python 3.12+
- CUDA-compatible GPU (recommended for training)
- Git

## Setup Instructions

1. **Clone Repository**:
   ```bash
   git clone https://github.com/bbaral-eng/Recurrent_Rainbow_DQN.git
   cd Recurrent_Rainbow_DQN
   ```

2. **Create Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training Recurrent DQN
```bash
jupyter notebook RL_Protocol_Extraction_Agents/RecurrentDQN.ipynb
```
- Configure hyperparameters in training cell
- LSTM parameters: 256 hidden units, 1 layer
- Action masking enabled by default
- Models saved to `models/RecurrentDQN_*.pkl`
- Also includes a real-world deployment interface (`RealWorldAgent`) for running the trained agent on live EMF readings instead of the simulator

### Training Recurrent PPO (LSTM)
```bash
jupyter notebook RL_Protocol_Extraction_Agents/RecurrentPPO.ipynb
```
- LSTM-based policy and value networks
- On-policy learning with 4 parallel environments
- Action masking via logit manipulation
- Models saved to `models/RecurrentPPO_*.pkl`

### Running Baselines
```bash
jupyter notebook benchmarking/baselines.ipynb
```
- Compare against Random, Greedy, Entropy-Weighted strategies
- Standard (non-recurrent) DQN and PPO agents

### Multi-Seed Training
```bash
jupyter notebook benchmarking/MultiSeed_Training.ipynb
```
- Trains each algorithm variant across 5 seeds for statistical comparison
- Outputs CSVs to `PlotsInPaper/MultiSeed/` (kept locally, not pushed to GitHub)

### Visualization
All notebooks include visualization cells that show:
- Action probability/Q-value heatmaps
- Action masking (red X markers for invalid actions)
- Belief entropy evolution
- Episode trajectories

## Key Features

### Belief-Guided Action Masking
- **DRQN**: Sets Q-values to `-∞` for invalid actions based on belief state
- **PPO**: Sets logits to `-∞` before softmax for invalid actions
- Ensures physically impossible actions are never selected
- Dramatically improves sample efficiency

### Bayesian Belief State
- Particle filter-based position estimation
- Temporal prediction with velocity constraints
- EMF measurement correction with Gaussian likelihood
- Entropy-based uncertainty quantification

### Custom Feature Extractor
- 3-layer CNN for voltage history processing (32→64→64 filters)
- MLP projection for scalar observations
- Shared feature extraction across actor/critic (PPO) or Q-network (DRQN)

### Environment Configuration
- Customizable grid layouts via `configuration/grid_configuration.json`
- Adjustable EMF field parameters (sigma, noise)
- Multiple protocol generation strategies
- Obstacle support

## File Structure

```
├── RL_Protocol_Extraction_Agents/  # Agents that reverse-engineer protocols - the main deliverable
│   ├── RecurrentDQN.ipynb          # DRQN implementation (LSTM) + real-world deployment interface
│   └── RecurrentPPO.ipynb          # PPO implementation (LSTM)
├── benchmarking/                   # Comparison points for the agents above
│   ├── baselines.ipynb             # Random / Greedy / Entropy-weighted / SB3 DQN & PPO
│   └── MultiSeed_Training.ipynb    # Multi-seed training runs for statistical comparison
├── future_DMFB_research/
│   └── closed_top_study.ipynb      # Closed-top DMFB feasibility study
├── environment_utils/              # DMFB environment and supporting modules
│   ├── Env.py                      # DMFB POMDP environment
│   ├── belief_state.py             # Bayesian state estimation
│   ├── protocol_creation.py        # Droplet path generation
│   ├── helper_functions.py         # Utility functions
│   ├── CustomFeatureExtractor.py   # CNN+MLP feature extractor
│   └── rendering.py                # Visualization utilities
├── configuration/
│   ├── config.py                   # Environment parameters + interactive grid editor
│   ├── grid_configuration.json     # Grid layout specification
│   └── data_collection/
│       ├── O_scope_data_collection.py  # Oscilloscope data capture
│       └── plot_O_Scope.py             # Oscilloscope data plotting
├── PlotsInPaper/                   # Figures and CSVs for the paper (local only, not pushed)
├── defense/                        # Reserved for a future defense algorithm
├── models/                         # Saved models and TensorBoard logs
├── O_Scope_Data/                   # EMF measurement data
├── actual_chip.jpeg                # DMFB chip photo used in figures
├── requirements.txt                # Python dependencies
```

## Configuration

### Environment Parameters (`configuration/config.py`)
```python
gridSize = 11        # grid size (gridSize x gridSize)
numDroplets = 1       # number of droplets
numAgents = 1         # number of agents (capped at 1)
sigma = 1.5           # EMF field spread
window_size = 600     # rendering window size
render_fps = 8        # render fps
k = 2                 # voltage_history depth divisor
threshold = 0.5       # belief threshold for action masking
```
Run `configuration/config.py` directly to open the interactive grid editor and regenerate `configuration/grid_configuration.json`.

### Model Hyperparameters

**DRQN**:
- Learning rate: 1e-4
- Replay buffer: 1000 episodes
- Batch size: 32 episodes
- Target network update: Every 500 steps
- Epsilon decay: Linear from 1.0 to 0.01

**RecurrentPPO (LSTM)**:
- Learning rate: 1e-4
- Parallel environments: 4
- Steps per rollout: 128
- PPO epochs: 4
- Minibatches: 4
- GAE lambda: 0.95
- Clip coefficient: 0.2

## Network Architectures

### DRQN
```
Input (Dict Observations)
    ↓
CustomFeatureExtractor (CNN + MLP)
    ↓
LSTM (256 hidden, 1 layer)
    ↓
Q-Value Head (MLP: 256→128→121 actions)
```

### RecurrentPPO
```
Input (Dict Observations)
    ↓
CustomFeatureExtractor (CNN + MLP)
    ↓
LSTM (256 hidden, 1 layer, shared)
    ↓
    ├─ Actor Head (MLP: 256→256→121 logits)
    └─ Critic Head (MLP: 256→256→1 value)
```

## Results

Models are evaluated on:
- **Normalized Return**: Reward per step (EMF accuracy)
- **Sample Efficiency**: Training steps to convergence
- **Action Mask Effectiveness**: Valid action selection rate
- **Belief Entropy Reduction**: Information gain per episode

TensorBoard logs available in `models/tensorboard/`:
```bash
tensorboard --logdir models/tensorboard
```

## Hardware Requirements

- **Minimum**: CPU with 8GB RAM (training will be slow)
- **Recommended**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: ~2GB for models and logs (excluding large data files)


## License

MIT License - See LICENSE file for details.
