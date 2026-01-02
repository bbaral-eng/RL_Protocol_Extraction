# Recurrent Reinforcement Learning for Digital Microfluidic Protocol Extraction

This repository implements recurrent reinforcement learning algorithms for protocol reverse-engineering on Digital Microfluidic Biochips (DMFBs). Agents learn to infer hidden droplet protocols by strategically probing electromagnetic field (EMF) measurements in a partially observable environment.

## Problem Statement

Agents operate in a Partially Observable Markov Decision Process (POMDP) with:
- **Spatial Ambiguity**: Single EMF readings correspond to multiple possible droplet positions
- **Temporal Dependencies**: Droplet motion has velocity constraints and physics-based dynamics
- **Hidden Protocol**: The true droplet path is unknown and must be reverse-engineered

## Implemented Algorithms

### 1. Recurrent Deep Q-Network (DRQN)
- **Architecture**: LSTM-based recurrent memory (256 hidden units)
- **Features**: Double DQN, Dueling Network, Prioritized Experience Replay
- **File**: `RecurrentDQN.ipynb`

### 2. Recurrent Proximal Policy Optimization (PPO-GTrXL)  
- **Architecture**: Gated Transformer-XL with multi-head attention
- **Features**: Policy gradient optimization with transformer-based memory
- **File**: `RecurrentPPO_GTrXL.ipynb`

### 3. Time Series Classification
- **Purpose**: Electrode identification from EMF signatures using dl-4-tsc ResNet
- **File**: `TSC_O_Scope_Data.py`

## Prerequisites

- Python 3.12
- CUDA-compatible GPU (recommended)
- Git

## Setup Instructions

1. **Clone Repository**:
   ```bash
   git clone https://github.com/bbaral-eng/Recurrent_Rainbow_DQN.git
   cd Recurrent_Rainbow_DQN
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **For Time Series Classification** (optional):
   ```bash
   git clone https://github.com/hfawaz/dl-4-tsc.git
   cd dl-4-tsc && pip install -r requirements.txt
   ```

## Usage

### Training Recurrent DQN
```bash
jupyter notebook RecurrentDQN.ipynb
```
- Modify hyperparameters in the configuration cell
- Execute all cells to train LSTM-based agent
- Results and models saved to `models/` directory

### Training Recurrent PPO with GTrXL
```bash
jupyter notebook RecurrentPPO_GTrXL.ipynb
```
- Configure transformer parameters (layers, heads, memory length)
- Execute training cells for transformer-based agent
- Tensorboard logs available in `models/tensorboard/`

### Time Series Classification
```bash
# Train classifier
python TSC_O_Scope_Data.py --train data.csv --output ./results --epochs 500

# Predict new samples
python TSC_O_Scope_Data.py --predict new_data.csv --model ./results
```

## Key Features

### Physics-Based Action Masking
- Gradient detachment for physically impossible actions
- Velocity constraints embedded in learning process
- Configurable spatial constraint windows (3x3, 5x5, 7x7)

### Belief State Management
- Bayesian filtering for droplet position estimation
- Temporal prediction and correction steps
- Entropy-based uncertainty quantification

### Environment Configuration
- Customizable grid layouts via `grid_configuration.json`
- Adjustable probe noise and physics parameters
- Multiple protocol generation strategies

## File Structure

```
├── RecurrentDQN.ipynb              # DRQN implementation
├── RecurrentPPO_GTrXL.ipynb        # PPO-GTrXL implementation  
├── TSC_O_Scope_Data.py            # Time series classification
├── Env.py                         # DMFB environment
├── belief_state.py                # Bayesian state estimation
├── protocol_creation.py           # Droplet path generation
├── helper_functions.py            # Utility functions
├── CustomFeatureExtractor.py      # Network architectures
├── config.py                      # Environment parameters
├── requirements.txt               # Python dependencies
└── models/                        # Saved models and logs
```

## Configuration

### Environment Parameters (`config.py`)
- Grid size and electrode positions
- EMF field parameters (sigma, threshold)
- Probe noise characteristics
- Episode length and termination conditions

### Model Hyperparameters
- **DRQN**: Learning rate, replay buffer size, epsilon decay
- **PPO-GTrXL**: Policy/value coefficients, transformer dimensions
- **Both**: Batch size, training steps, evaluation frequency

## Results

Models are evaluated on:
- **Protocol Accuracy**: Percentage of correctly inferred droplet paths
- **Sample Efficiency**: Training steps required for convergence
- **Generalization**: Performance on unseen grid configurations

Trained models and tensorboard logs are saved in `models/` for analysis.

## Hardware Requirements

- **Minimum**: CPU with 8GB RAM
- **Recommended**: NVIDIA GPU with 8GB+ VRAM for transformer training
- **Storage**: 2GB for models and training logs

## Citation

```bibtex
@article{baral2024recurrent,
  title={Recurrent Reinforcement Learning for Digital Microfluidic Protocol Extraction},
  author={Baral, Biplove and [Additional Authors]},
  journal={[Journal Name]},
  year={2024}
}
```

## License

MIT License - See LICENSE file for details.
