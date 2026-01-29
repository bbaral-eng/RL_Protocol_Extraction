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
- **File**: `RecurrentDQN.ipynb`

### 2. Recurrent Proximal Policy Optimization (PPO)
- **Architecture**: LSTM-based recurrent memory (256 hidden units, 1 layer)
- **Features**: On-policy learning with GAE, separate actor-critic heads
- **Action Masking**: Belief-guided masking via logit manipulation
- **File**: `RecurrentPPO.ipynb`

### 3. Recurrent PPO with Transformer-XL (Alternative)
- **Architecture**: Gated Transformer-XL with multi-head attention (2 layers, 4 heads)
- **Features**: Attention-based memory with 32-timestep sliding window
- **File**: `RecurrentPPO_TrXL.ipynb`

### 4. Baseline Algorithms
- **Methods**: Random, Greedy, Entropy-Weighted, Standard DQN, Standard PPO
- **File**: `baselines.ipynb`

### 5. Time Series Classification
- **Purpose**: Electrode identification from EMF signatures using dl-4-tsc ResNet
- **File**: `droplet_classify.ipynb`

## Prerequisites

- Python 3.12+
- CUDA-compatible GPU (recommended for training)
- Git

## Setup Instructions

1. **Clone Repository**:
   ```bash
   git clone https://github.com/[REPOSITORY_URL].git
   cd [REPOSITORY_NAME]
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
jupyter notebook RecurrentDQN.ipynb
```
- Configure hyperparameters in training cell
- LSTM parameters: 256 hidden units, 1 layer
- Action masking enabled by default
- Models saved to `models/RecurrentDQN_*.pkl`

### Training Recurrent PPO (LSTM)
```bash
jupyter notebook RecurrentPPO.ipynb
```
- LSTM-based policy and value networks
- On-policy learning with 4 parallel environments
- Action masking via logit manipulation
- Models saved to `models/RecurrentPPO_*.pkl`

### Running Baselines
```bash
jupyter notebook baselines.ipynb
```
- Compare against Random, Greedy, Entropy-Weighted strategies
- Standard (non-recurrent) DQN and PPO agents

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
- Customizable grid layouts via `grid_configuration.json`
- Adjustable EMF field parameters (sigma, noise)
- Multiple protocol generation strategies
- Obstacle support

## File Structure

```
├── RecurrentDQN.ipynb              # DRQN implementation (LSTM)
├── RecurrentPPO.ipynb              # PPO implementation (LSTM)
├── RecurrentPPO_TrXL.ipynb         # PPO implementation (Transformer-XL)
├── baselines.ipynb                 # Baseline comparisons
├── droplet_classify.ipynb          # Time series classification
├── Env.py                          # DMFB POMDP environment
├── belief_state.py                 # Bayesian state estimation
├── protocol_creation.py            # Droplet path generation
├── helper_functions.py             # Utility functions
├── CustomFeatureExtractor.py       # CNN+MLP feature extractor
├── rendering.py                    # Visualization utilities
├── config.py                       # Environment parameters
├── grid_configuration.json         # Grid layout specification
├── requirements.txt                # Python dependencies
├── models/                         # Saved models and TensorBoard logs
└── O_Scope_Data/                   # EMF measurement data
```

## Configuration

### Environment Parameters (`config.py`)
```python
GRID_SIZE = 18              # Grid dimensions (18x18)
SIGMA = 3.0                 # EMF field spread
EMF_THRESHOLD = 0.01        # Measurement sensitivity
PROBE_NOISE_STD = 0.05      # Gaussian noise std
MAX_EPISODE_LENGTH = 100    # Episode termination
```

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
