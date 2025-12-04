# Deep Recurrent Q-Network (DRQN) for Droplet Protocol Reverse Engineering# Recurrent DQN (R2D1)



This repository implements Deep Recurrent Q-Networks (DRQN) with support for both **LSTM** and **Gated Transformer-XL (GTrXL)** architectures for learning to track and predict droplet movements on Digital Microfluidic Biochips (DMFBs). The agent learns to reverse-engineer droplet protocols by observing electromagnetic field (EMF) measurements through a partially observable environment.This repository contains an implementation of a Recurrent DQN agent based on the R2D1 (Recurrent Replay Distributed DQN) architecture. This is a streamlined variant that combines key improvements to vanilla DQN without the full Rainbow suite:



## 🔬 Problem Overview**Core Components:**

- **Double DQN**: Reduces overestimation bias by using the online network to select actions and the target network to evaluate them

The agent faces a **Partially Observable Markov Decision Process (POMDP)** where:- **Prioritized Experience Replay (PER)**: Samples important transitions more frequently based on TD-error

- **Spatial Ambiguity**: Single EMF reading at one probe location can correspond to multiple possible droplet positions- **Dueling Architecture**: Separates state value and action advantage streams for better learning

- **Temporal Dependencies**: Droplet movement has velocity constraints - it cannot teleport- **N-step Learning**: Uses multi-step returns to reduce variance and improve convergence

- **Physics Constraints**: EMF strength decreases radially from the droplet's actual position- **Custom Feature Extractor**: Handles dictionary observations from the environment



The agent must learn to:The code is structured to be run in a Jupyter notebook environment with specific dependencies and Python version requirements.

1. **Estimate velocity** from EMF history (temporal abstraction)

2. **Predict droplet location** given ambiguous single-point measurements## Prerequisites

3. **Respect physics** by understanding droplet motion constraints

- Python 3.12

## 🚀 Key Features- Git (for cloning the repository)



### Architecture Options## Setup Instructions

- **LSTM (Default)**: Traditional recurrent memory with 256 hidden units - fast, proven baseline

- **GTrXL**: Transformer-based memory with multi-head attention for longer-term dependencies1. Clone the repository:

  - Configurable layers (2-4), attention heads (4-8), memory length (15-25)   ```bash

  - Optimized hyperparameters for stable transformer training   git clone https://github.com/bbaral-eng/Recurrent_Rainbow_DQN.git

   cd RecurrentDQN

### Physics-Based Action Masking   ```

Novel implementation that embeds physics constraints into the learning process:

- **Gradient Detachment**: Invalid actions (physically impossible given velocity) have gradients detached during training2. Create a virtual environment:

- **Forces Learning**: Network learns velocity and direction rather than memorizing all Q-values   ```bash

- **Configurable Radius**: 3x3, 5x5, or 7x7 spatial constraint windows   python3.12 -m venv venv

   ```

### Episode-Based Learning

- Stores complete episodes (not single transitions) for sequential learning3. Activate the virtual environment:

- Supports both architectures through unified replay buffer   - On macOS/Linux:

- Efficient episode sampling for recurrent training     ```bash

     source venv/bin/activate

## 📋 Prerequisites     ```

   - On Windows:

- Python 3.12     ```bash

- macOS/Linux (tested on macOS)     .\venv\Scripts\activate

- Git     ```



## 🛠️ Setup Instructions4. Install dependencies:

   ```bash

1. **Clone the repository:**   pip install -r requirements.txt

   ```bash   ```

   git clone https://github.com/bbaral-eng/Recurrent_Rainbow_DQN.git

   cd official_repo## Running the Code

   ```

1. Ensure your virtual environment is activated (you should see `(venv)` in your terminal prompt)

2. **Create a virtual environment:**

   ```bash2. Launch Jupyter Notebook:

   python3.12 -m venv RRDQNvenv   ```bash

   ```   jupyter notebook

   ```

3. **Activate the virtual environment:**

   - macOS/Linux:3. Open the `RecurrentDQN.ipynb` notebook

     ```bash

     source RRDQNvenv/bin/activate4. Run all cells in the notebook sequentially from top to bottom

     ```

   - Windows:**Alternative: Run via CLI (for hyperparameter tuning)**

     ```bash

     .\RRDQNvenv\Scripts\activateFor hyperparameter sweeps, use the CLI script with specific hyperparameters:

     ```

```bash

4. **Install dependencies:**# Activate your venv

   ```bashsource venv/bin/activate

   pip install -r requirements.txt

   ```# Run a single experiment

python train_coreweave.py \

## 🎮 Running the Code  --lr 2.5e-4 \

  --gamma 0.99 \

### Option 1: Jupyter Notebook (Recommended)  --n_step 1 \

  --alpha 0.1 \

1. Ensure virtual environment is activated:  --beta 0.6 \

   ```bash  --prior_eps 1e-6 \

   source RRDQNvenv/bin/activate  --seed 42 \

   ```  --total_timesteps 40000

```

2. Launch Jupyter:

   ```bashFor Kubernetes/CoreWeave distributed runs:

   jupyter notebook

   ``````bash

# Generate job manifests

3. Open `RecurrentDQN.ipynb`python launch_coreweave_experiments.py --generate-only



4. Follow the notebook sections:# Apply to cluster (requires kubeconfig)

   - **Section 1**: Import librarieskubectl apply -f k8s_jobs/

   - **Section 2**: Train LSTM agent (40k steps)```

   - **Section 3**: Train GTrXL agent (40k steps, optimized hyperparameters)

   - **Section 4**: Visualize Q-values and agent behavior## Monitoring / TensorBoard

   - **Section 5**: Cross-episode learning experiments

We log training and evaluation metrics to TensorBoard so you can inspect training progress, compare runs, and export figures. Logs are written using the PyTorch TensorBoard writer and stored under the project folder:

### Option 2: Direct Training (LSTM)

- `models/DQN_tensorboard/<run_name>/`

```python

from Env import VoltageDistributionEnvWhen you create the agent you can pick a run name (for example `tb_run_name="experiment_001"`) and that will create a subfolder inside `models/DQN_tensorboard`. If no run name is provided the code will create a timestamped folder for you.

from CustomFeatureExtractor import CustomFeatureExtractor

Common SB3-style scalars that are logged by default:

# Get action space dimension

env = VoltageDistributionEnv()- `rollout/ep_len_mean` — mean episode length

action_dim = env.action_space.n- `rollout/ep_rew_mean` — mean episode reward

- `rollout/exploration_rate` — epsilon / exploration rate

# Create LSTM agent- `time/fps` — frames-per-second (training speed)

agent = DRQNagent(- `train/learning_rate` — current optimizer learning rate

    env=lambda: VoltageDistributionEnv(),- `train/loss` — training loss

    feature_extractor=lambda obs_space: CustomFeatureExtractor(obs_space),

    action_dim=action_dim,Note: When starting TensorBoard you may see:

    lstm_hidden_size=256,

    gamma=0.99,```

    lr=1e-4,TensorFlow installation not found - running with reduced feature set.

    batch_size=32,```

    buffer_size=1_000,

    epsilon_start=1.0,This is normal — TensorBoard can run without TensorFlow, but installing TensorFlow will enable a few extra visualization features. To install TensorFlow (optional):

    epsilon_end=0.001,

    epsilon_decay=5_000,```bash

    target_update=250,pip install tensorflow

    training_steps=40_000,```

    tensorboard_dir='models/DQN_tensorboard',

    run_name='LSTM_40k',Start TensorBoard from the repository root and point it at `models/DQN_tensorboard`:

    use_action_masking=True,

    action_mask_size=5```bash

)tensorboard --logdir=models/DQN_tensorboard

```

# Train

agent.train()Open http://localhost:6006/ in your browser and choose the run folder you want to inspect.



# Save## Important Notes

agent.save_complete('models/LSTM_40k.pkl')

```- Make sure to use Python 3.12 specifically, as the code has been tested and verified with this version

- The notebook must be run in sequential order as later cells depend on the execution of previous cells

### Option 3: GTrXL Training (Optimized)- Do not skip any cells as this may cause dependency or initialization issues

- If you encounter any memory issues, you may need to restart the kernel and run all cells again

```python- For distributed training on CoreWeave, ensure Docker and kubectl are configured properly

# Create GTrXL agent with optimized hyperparameters

agent = DRQNagent(## Hyperparameter Tuning

    env=lambda: VoltageDistributionEnv(),

    feature_extractor=lambda obs_space: CustomFeatureExtractor(obs_space),The repository supports both local and distributed hyperparameter optimization:

    action_dim=action_dim,

    use_gtrxl=True,**Local tuning (small grid):**

    gtrxl_n_layers=2,```bash

    gtrxl_n_heads=4,source venv/bin/activate

    gtrxl_d_head_inner=32,python train_coreweave.py --lr 1e-4 --gamma 0.99 --n_step 1 --alpha 0.0 --beta 0.6 --prior_eps 1e-6 --seed 42 --total_timesteps 40000

    gtrxl_d_ff_inner=256,```

    gtrxl_mem_len=15,

    gtrxl_dropout=0.05,**Distributed tuning (CoreWeave):**

    lr=5e-5,              # Lower LR for transformer stability1. Update `launch_coreweave_experiments.py` with your Docker image and namespace

    batch_size=16,        # Smaller batch for better gradients2. Generate job YAMLs: `python launch_coreweave_experiments.py --generate-only`

    target_update=100,    # More frequent updates3. Deploy: `kubectl apply -f k8s_jobs/`

    training_steps=40_000,4. Monitor results via W&B dashboard

    tensorboard_dir='models/DQN_tensorboard',

    run_name='GTrXL_40k_optimized',## Troubleshooting

    use_action_masking=True,

    action_mask_size=5If you encounter any issues:

)1. Verify that you're using Python 3.12

2. Confirm all dependencies are installed correctly

agent.train()3. Try restarting the Jupyter kernel

agent.save_complete('models/GTrXL_40k_optimized.pkl')4. Ensure all cells are run in order from top to bottom

```

## Project Structure

## 📊 Monitoring with TensorBoard

- `RecurrentDQN.ipynb`: Main notebook containing the R2D1 implementation

Training metrics are logged to TensorBoard:- `train_coreweave.py`: CLI script for single or batch hyperparameter experiments

- `launch_coreweave_experiments.py`: Kubernetes job launcher for distributed training

```bash- `Dockerfile`: Container specification for CoreWeave deployment

tensorboard --logdir=models/DQN_tensorboard- `CustomFeatureExtractor.py`: Environment-specific feature extraction

```- `Env.py`: Custom training environment

- `segment_tree.py`: Efficient segment tree for prioritized replay

Open http://localhost:6006/ in your browser.- `requirements.txt`: Python dependencies

- `k8s_jobs/`: Generated Kubernetes job manifests (created by launch script)

### Key Metrics:

- **`rollout/normalized_return`**: Reward per step (target: ~0.8-1.0)## References

- **`rollout/ep_rew_mean`**: Average episode return (last 100 episodes)

- **`rollout/epsilon`**: Exploration rate decayThis implementation is based on:

- **`train/loss`**: TD-error loss (should decrease and stabilize)- Deep Recurrent Q-Learning for Partially Observable MDPs (Hausknecht & Stone, 2017)

- **`rollout/ep_len_mean`**: Average episode length (should decrease as agent improves)- Recurrent Experience Replay in Distributed Reinforcement Learning (Kapturowski et al., 2019)

- Implementation inspiration: https://github.com/Curt-Park/rainbow-is-all-you-need

### Expected Performance:

- **LSTM**: Reaches ~0.75-0.85 normalized return by 40k steps
- **GTrXL**: Reaches ~0.70-0.80 normalized return (slower initial learning but better long-term memory)

## 🧪 Action Masking: Physics-Based Learning

The key innovation in this implementation is **physics-guided gradient masking**:

### Standard Approach (What Others Do):
```python
# Compute Q-values for all actions
q_values = network(obs)  # Shape: [396]

# Mask invalid actions during selection
q_values[invalid] = -inf
action = argmax(q_values)

# Problem: Network still trains on all 396 actions
loss.backward()  # Updates all Q-values
```

### Our Approach (Physics-Based):
```python
# Compute Q-values for all actions
q_values = network(obs)  # Shape: [396]

# DETACH gradients for invalid actions
q_values[invalid] = q_values[invalid].detach()

# Mask during selection
q_values[invalid] = -inf
action = argmax(q_values)

# Network only learns about valid actions
loss.backward()  # Gradients only flow through valid Q-values
```

**Result**: Network learns that actions far from the droplet's predicted trajectory are physically impossible, forcing it to learn velocity estimation and temporal patterns rather than memorizing a lookup table.

Enable with: `use_action_masking=True, action_mask_size=5` (5x5 spatial window)

## 📁 Project Structure

```
official_repo/
├── RecurrentDQN.ipynb          # Main notebook (LSTM + GTrXL training)
├── RecurrentPPO.ipynb          # PPO TrXL implementation (WIP)
├── Env.py                      # Custom DMFB environment
├── CustomFeatureExtractor.py  # Feature extraction for dict observations
├── GTrXL.py                    # Gated Transformer-XL implementation
├── config.py                   # Environment configuration
├── grid_configuration.json     # Electrode grid layout
├── helper_functions.py         # Voltage calculations, observations
├── protocol_creation.py        # Droplet path generation
├── rendering.py                # Environment visualization
├── requirements.txt            # Python dependencies
├── models/
│   ├── DQN_tensorboard/        # TensorBoard logs
│   │   ├── LSTM_40k/
│   │   └── GTrXL_40k_optimized/
│   ├── LSTM_40k.pkl            # Saved LSTM agent
│   └── GTrXL_40k.pkl           # Saved GTrXL agent
└── RRDQNvenv/                  # Virtual environment
```

## 🔧 Hyperparameter Comparison

| Parameter | LSTM | GTrXL (Optimized) | Why Different? |
|-----------|------|-------------------|----------------|
| Learning Rate | 1e-4 | 5e-5 | Transformers need gentler optimization |
| Batch Size | 32 | 16 | Smaller batches = better gradient signal |
| Target Update | 250 | 100 | More frequent updates stabilize attention |
| Gradient Clip | 10.0 | 1.0 | Prevent gradient explosion in transformers |
| Memory | 256 units | 15 timesteps | Different memory mechanisms |
| Dropout | N/A | 0.05 | Light regularization for transformers |

## 📈 Visualization & Analysis

The notebook includes tools to:
1. **Q-value Heatmaps**: Visualize agent's action preferences over the grid
2. **Episode Playback**: Watch agent behavior with live Q-value updates
3. **Action Masking Visualization**: See which actions are valid/invalid
4. **Cross-Episode Learning**: Test agent's ability to adapt online

## 🐛 Troubleshooting

### Console Output Shows Large Gaps
✅ **Fixed**: Now prints every 500 steps instead of every 100 episodes

### GTrXL Not Learning
✅ **Fixed**: Use optimized hyperparameters (lower LR, smaller batch, more frequent updates)

### Action Masking Not Working
- Ensure `use_action_masking=True` when creating agent
- Check that masks are being stored with transitions (6-tuple format)
- Verify gradient detachment is happening in `train_step()`

### Memory Issues
- Reduce `buffer_size` from 1000 to 500
- Decrease `batch_size` (especially for GTrXL)
- Restart kernel and clear output cells

## 🎓 References

### Core Papers:
1. **Deep Recurrent Q-Learning for Partially Observable MDPs**
   - Hausknecht & Stone, 2017
   - [arXiv:1507.06527](https://arxiv.org/abs/1507.06527)

2. **Stabilizing Transformers for Reinforcement Learning**
   - Parisotto et al., 2019
   - [arXiv:1910.06764](https://arxiv.org/abs/1910.06764)

3. **Gated Transformer-XL for Reinforcement Learning**
   - Original GTrXL architecture for RL

### Implementation Inspirations:
- CleanRL's PPO TrXL implementation
- Stable-Baselines3 design patterns
- Memory Gym benchmark environments

## 💡 Key Innovations

1. **Physics-Based Action Masking**: Embeds droplet physics into learning through gradient detachment
2. **Unified LSTM/GTrXL Architecture**: Single codebase supports both with architecture flag
3. **Episode-Based Replay**: Preserves temporal structure for recurrent learning
4. **Optimized GTrXL Training**: Carefully tuned hyperparameters for stable transformer learning
5. **Comprehensive Visualization**: Q-value heatmaps with action masking overlays

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{baral2025drqn_dmfb,
  author = {Baral, Biplove},
  title = {Deep Recurrent Q-Networks for Droplet Protocol Reverse Engineering},
  year = {2025},
  url = {https://github.com/bbaral-eng/Recurrent_Rainbow_DQN}
}
```

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the author.

## 📄 License

MIT License - see LICENSE file for details

---

**Note**: This implementation focuses on DRQN with LSTM/GTrXL. The repository name references "Rainbow DQN" from earlier work but the current implementation is pure DRQN with recurrent architectures optimized for partially observable environments.
