import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym

with open('../configuration/grid_configuration.json', 'r') as f:
    config = json.load(f)

grid_size = config['gridSize']
electrode_positions = config['electrode_positions']
obstacles = config['obstacles']
obstacle_set = set(tuple(obs) for obs in obstacles)

all_grid_positions = [(i, j) for i in range(grid_size) for j in range(grid_size)]

def extract_qvalues(model, obs_dict, hidden=None):

    """Extract Q-values for given observation using the trained model"""

    with torch.no_grad():
        obs_batch = {k: torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float()
                     for k, v in obs_dict.items()}
        q_vals, new_hidden = model.online_net(obs_batch, hidden)
        q_values = q_vals.squeeze()

        return q_values.cpu().numpy(), new_hidden

def plot_qvalues_heatmap(q_values, action_mask=None, model=None):

    """
    Plot heatmap of Q-values over the full grid action space.
    Masked (invalid) actions are shown as white (NaN) cells with red X markers.
    Electrodes are marked with white borders.
    """

    plt.figure(figsize=(10, 8))
    q_grid = np.full((grid_size, grid_size), np.nan)

    # Action space is full grid: action idx = row * grid_size + col
    for idx in range(grid_size * grid_size):
        row = idx // grid_size
        col = idx % grid_size
        # Mask invalid actions for visualization
        if action_mask is not None and action_mask[idx] == 0:
            q_grid[row, col] = np.nan
        else:
            q_grid[row, col] = q_values[idx]

    # Find best valid action (considering mask if provided)
    if action_mask is not None:
        valid_indices = np.where(action_mask == 1)[0]
        if len(valid_indices) > 0:
            valid_q_values = q_values[valid_indices]
            max_q_idx = valid_indices[np.argmax(valid_q_values)]
        else:
            max_q_idx = np.argmax(q_values)
    else:
        max_q_idx = np.argmax(q_values)

    # Convert action index to row/col
    max_row = max_q_idx // grid_size
    max_col = max_q_idx % grid_size

    # Set white background for NaN values
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='white')

    plt.imshow(q_grid, cmap=cmap, interpolation='nearest', origin='upper')
    plt.colorbar(label='Q-value')

    # Highlight masked (invalid) actions with red X if action masking is used
    if action_mask is not None:
        for idx in range(grid_size * grid_size):
            if action_mask[idx] == 0:
                row = idx // grid_size
                col = idx % grid_size
                plt.plot(col, row, 'rx', markersize=8, markeredgewidth=1.5)

    # Draw white borders around electrodes to show their positions
    for electrode in electrode_positions:
        electrode_row, electrode_col = electrode[0], electrode[1]
        rect = plt.Rectangle((electrode_col - 0.5, electrode_row - 0.5), 1, 1,
                             linewidth=2, edgecolor='white', facecolor='none')
        plt.gca().add_patch(rect)

    # Red rectangle around best action
    rect = plt.Rectangle((max_col - 0.5, max_row - 0.5), 1, 1, linewidth=3, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect)

    title = 'Q-values Heatmap over Full Grid'
    if action_mask is not None:
        num_valid = int(action_mask.sum())
        title += f' (Masking: {num_valid}/{len(action_mask)} valid)'
    plt.title(title)
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()

class MultipassEnvWrapper(gym.Wrapper):
    """
    Wrapper ensures same protocol is repeated across passes.

    - Pass 1: Let env generate protocol, store _droplet_path
    - Pass 2+: Override _droplet_path with previously generated protocol
    - Injects refined masks from PassBasedMasking using three-stage refinement

    """

    def __init__(self, env, pass_based_masking, pass_num, stored_protocol=None):
        super().__init__(env)
        self.pass_based_masking = pass_based_masking
        self.pass_num = pass_num
        self.stored_protocol = stored_protocol
        self.timestep = 0
        self.grid_size = env.size

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # For passes greater than 1, override droplet path with stored protocol
        if self.stored_protocol is not None and self.pass_num > 1:
            self.env._droplet_path = self.stored_protocol.copy()
            self.env._droplet_path_index = 0
            self.env._droplet_location = self.env._droplet_path[0]
            self.env.end_pos = self.env._droplet_path[-1]
            self.env.length = len(self.env._droplet_path)

            # Recalculate voltage grid for starting position
            from helper_functions import voltage_grid as calc_voltage_grid
            self.env._voltage_grid = calc_voltage_grid(
                self.env._droplet_location,
                self.env.belief.sigma,
                obstacles,
                self.grid_size
            )

            # Get refined action mask for timestep 0 using previous pass data
            current_mask = info['action_mask']
            refined_mask = self.pass_based_masking.get_refined_mask(timestep=0, current_belief_mask=current_mask)
            info['action_mask'] = refined_mask

        self.timestep = 0
        self.pass_based_masking.start_new_episode()

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Extract data for recording
        probe_position = (action // self.grid_size, action % self.grid_size)
        emf = obs['voltage'][0]
        corrected_belief = info.get('filtered_belief', None)

        if corrected_belief is not None:
            self.pass_based_masking.record_step(self.timestep, probe_position, emf, corrected_belief)

        self.timestep += 1

        # For Pass 2+, override action mask with refined mask
        if self.pass_num > 1:
            current_mask = info.get('action_mask', None)
            if current_mask is not None:
                refined_mask = self.pass_based_masking.get_refined_mask(self.timestep, current_mask)
                info['action_mask'] = refined_mask

        if terminated or truncated:
            self.pass_based_masking.end_episode()

        return obs, reward, terminated, truncated, info

    def get_stored_protocol(self):
        """Get droplet path to reuse in subsequent passes."""
        return self.env._droplet_path.copy()

class PassBasedMasking:
    """
    Three-stage belief refinement for multipass protocol reconstruction.

    Stage 1: Historical Belief Constraint
        - Use corrected belief from previous pass at each timestep
        - If belief was 0 at a location, it stays 0 for current pass as well

    Stage 2: Global Smoothing
        - Propagate constraints from all known locations both forward and backward in time
        - Forward propagation: reachable from known past locations
        - Backward propagation: can reach known future locations

    Stage 3: Ensure Decrease in Valid Locations
        - Number of valid locations never increases pass over pass
        - If current refinement has more locations than previous pass, use previous

    Key principle: Belief only shrinks or stays same, never expands.
    """

    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.num_states = grid_size * grid_size


        self.current_trajectory = []   # Current
        self.previous_trajectory = []  # Previous

        self.neighbor_map = self._build_neighbor_map()

    def _build_neighbor_map(self):
        """Build adjacency graph for transition model (N/E/S/W moves)."""
        neighbor_map = {}
        for idx in range(self.num_states):
            row, col = self._idx_to_pos(idx)
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    neighbors.append(nr * self.grid_size + nc)
            neighbor_map[idx] = neighbors
        return neighbor_map

    def _get_neighbors(self, pos):
        """Get 4-connected neighbors (N/E/S/W) of a position."""
        row, col = pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                neighbors.append((nr, nc))
        return neighbors

    def _pos_to_idx(self, pos):
        """Convert (row, col) to action index."""
        return pos[0] * self.grid_size + pos[1]

    def _idx_to_pos(self, idx):
        """Convert action index to (row, col)."""
        return (idx // self.grid_size, idx % self.grid_size)

    def record_step(self, timestep, probe_position, emf, corrected_belief):
        """
        Record a single step's data during episode.

        Args:
            timestep: Current timestep
            probe_position: (row, col) tuple of probe location
            emf: EMF reading at probe position
            corrected_belief: Full belief distribution AFTER correction step (np.ndarray)
        """
        self.current_trajectory.append({
            't': timestep,
            'pos': probe_position,
            'emf': emf,
            'belief': corrected_belief.copy()
        })

    def start_new_episode(self):
        """Move current trajectory to previous, start fresh recording."""
        self.previous_trajectory = self.current_trajectory
        self.current_trajectory = []

    def end_episode(self):
        """Mark end of current episode (placeholder for future use)."""
        pass

    def get_refined_mask(self, timestep, current_belief_mask):
        """
        Compute refined mask using three-stage refinement.

        Args:
            timestep: Current timestep
            current_belief_mask: Mask from current Bayesian belief (from env), shape (num_states,)

        Returns:
            Refined action mask (np.ndarray of shape (num_states,))
        """

        if len(self.previous_trajectory) == 0:
            return current_belief_mask

        # ensure current mask is 1D
        current_belief_mask = np.array(current_belief_mask).flatten()

        traj_by_t = {step['t']: step for step in self.previous_trajectory}

        if timestep not in traj_by_t:
            return current_belief_mask

        # get historical belief and flatten to 1D if needed
        historical_belief = traj_by_t[timestep]['belief']
        historical_belief = np.array(historical_belief).flatten()

        historical_threshold = (1.0 / self.num_states) * 0.1  # Matches env threshold
        historical_mask = (historical_belief >= historical_threshold).astype(np.int32)

        refined_mask = current_belief_mask * historical_mask

        known_locations = {}
        for step in self.previous_trajectory:
            if step['emf'] >= 0.999:
                known_locations[step['t']] = self._pos_to_idx(step['pos'])

        smoothing_mask = self._global_smoothing(timestep, known_locations)

        refined_mask = refined_mask * smoothing_mask


        refined_count = refined_mask.sum()
        historical_count = historical_mask.sum()

        if refined_count > historical_count:
            refined_mask = historical_mask.copy()

        if refined_mask.sum() == 0:
            refined_mask = historical_mask.copy()
            if refined_mask.sum() == 0:
                refined_mask = current_belief_mask.copy()

        return refined_mask.astype(np.int32)

    def _global_smoothing(self, timestep, known_locations):

        """
        Apply global smoothing using transition model.

        For each known location (EMF=1), propagate reachability constraints:
        - Forward: cells reachable from past known location
        - Backward: cells that can reach future known locations

        Args:
            timestep: Target timestep to compute mask for
            known_locations: Dict {timestep: position_idx} for EMF=1 locations

        Returns:
            Smoothing constraint mask (np.ndarray)
        """
        if len(known_locations) == 0:
            return np.ones(self.num_states, dtype=np.int32)

        forward_mask = np.ones(self.num_states, dtype=np.int32)
        backward_mask = np.ones(self.num_states, dtype=np.int32)

        past_known = [(t, idx) for t, idx in known_locations.items() if t < timestep]

        if past_known:
            closest_past_t, closest_past_idx = max(past_known, key=lambda x: x[0])
            steps_forward = timestep - closest_past_t

            forward_mask = self._propagate_n_steps({closest_past_idx}, steps_forward)

        future_known = [(t, idx) for t, idx in known_locations.items() if t > timestep]

        if future_known:
            closest_future_t, closest_future_idx = min(future_known, key=lambda x: x[0])
            steps_backward = closest_future_t - timestep

            backward_mask = self._propagate_n_steps({closest_future_idx}, steps_backward)

        if timestep in known_locations:
            exact_mask = np.zeros(self.num_states, dtype=np.int32)
            exact_mask[known_locations[timestep]] = 1
            return exact_mask

        smoothing_mask = forward_mask * backward_mask

        return smoothing_mask

    def _propagate_n_steps(self, start_positions, n_steps):
        """
        Compute all cells reachable from start_positions in exactly n_steps.

        Uses BFS-style propagation through transition model.

        Args:
            start_positions: Set of starting position indices
            n_steps: Number of transition steps

        Returns:
            Mask of reachable positions (np.ndarray)
        """
        if n_steps == 0:
            mask = np.zeros(self.num_states, dtype=np.int32)
            for idx in start_positions:
                mask[idx] = 1
            return mask

        current_set = start_positions.copy()

        for _ in range(n_steps):
            next_set = set()
            for idx in current_set:
                for neighbor_idx in self.neighbor_map[idx]:
                    next_set.add(neighbor_idx)
            current_set = next_set

        # Convert to mask
        mask = np.zeros(self.num_states, dtype=np.int32)
        for idx in current_set:
            mask[idx] = 1

        return mask

def run_multipass_protocol(agent, num_passes=3, render=False, render_delay=0.5):
    """
    Run same protocol multiple times with iterative mask refinement.

    Uses PassBasedMasking with three-stage refinement:
    1. Historical belief constraint
    2. Global smoothing
    3. belief space decrease constraint

    Args:
        agent: Trained DRQNagent
        num_passes: Number of passes (default 3)
        render: Whether to render environment
        render_delay: Delay between steps when rendering

    Returns:
        results: List of dicts per pass with reward, steps, mask_sizes, entropy
    """
    from IPython.display import clear_output
    from Env import VoltageDistributionEnv

    # Load grid config
    with open('../configuration/grid_configuration.json', 'r') as f:
        grid_config = json.load(f)
    grid_size = grid_config['gridSize']

    # Shared masking logic across passes
    pass_based_masking = PassBasedMasking(grid_size)
    results = []
    stored_protocol = None

    for pass_num in range(1, num_passes + 1):

        # Create wrapped environment
        base_env = VoltageDistributionEnv(render_mode='human' if render else None)
        env = MultipassEnvWrapper(base_env, pass_based_masking, pass_num, stored_protocol)

        # Reset
        obs, info = env.reset()
        hidden = None  # LSTM hidden state

        # Store protocol from first pass
        if pass_num == 1:
            stored_protocol = env.get_stored_protocol()

        # Episode loop
        done = False
        total_reward = 0
        step_num = 0
        mask_sizes = []
        entropy_values = []  # Track entropy at each step

        while not done:
            if render:
                clear_output(wait=True)

            # Get action mask
            action_mask = info.get('action_mask', None)
            if action_mask is not None:
                mask_sizes.append(int(np.sum(action_mask)))

            # Track belief entropy
            current_entropy = env.env.belief.get_state_entropy()
            entropy_values.append(current_entropy)

            # Get Q-values from agent
            q_values, hidden = extract_qvalues(agent, obs, hidden)

            # Select action (greedy over valid actions)
            if action_mask is not None:
                valid_indices = np.where(np.array(action_mask) == 1)[0]
                if len(valid_indices) > 0:
                    action = valid_indices[np.argmax(q_values[valid_indices])]
                else:
                    action = np.argmax(q_values)
            else:
                action = np.argmax(q_values)

            # Render
            if render:
                env.render()
                pos = (action // grid_size, action % grid_size)
                print(f"Pass {pass_num} | Step {step_num} | Pos: {pos} | EMF: {obs['voltage'][0]:.3f} | Mask: {mask_sizes[-1] if mask_sizes else 'N/A'} | Entropy: {current_entropy:.3f}")
                plot_qvalues_heatmap(q_values, action_mask=action_mask, model=agent)
                time.sleep(render_delay)

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_num += 1

        # Compute metrics
        avg_mask = np.mean(mask_sizes) if mask_sizes else 0
        avg_entropy = np.mean(entropy_values) if entropy_values else 0
        final_entropy = entropy_values[-1] if entropy_values else 0
        normalized_return = total_reward / len(mask_sizes) if mask_sizes else 0

        result = {
            'pass_num': pass_num,
            'reward': total_reward,
            'normalized_return': normalized_return,
            'steps': len(mask_sizes),
            'mask_sizes': mask_sizes,
            'avg_mask_size': avg_mask,
            'entropy_values': entropy_values,
            'avg_entropy': avg_entropy,
            'final_entropy': final_entropy
        }
        results.append(result)

        env.close()

    # Summary table
    print("="*65)
    print(f"{'Pass':<6} {'Reward':<10} {'Norm Return':<12} {'Avg Entropy':<12} {'Avg Mask':<10} {'Steps':<8}")
    print("-"*65)
    for r in results:
        print(f"{r['pass_num']:<6} {r['reward']:<10.2f} {r['normalized_return']:<12.3f} {r['avg_entropy']:<12.3f} {r['avg_mask_size']:<10.1f} {r['steps']:<8}")

    return results, stored_protocol
