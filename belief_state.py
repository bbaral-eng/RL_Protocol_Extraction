import numpy as np 

class BeliefStateMap:

    """
    Bayesian belief tracker over electrodes

    The belief represents P(x_t | O_{1:t}), where x_t is the droplet location
    and O_t is the observations collected by the probe.

    """

    def __init__(self, electrode_positions, sigma, threshold): 

        """
        Args:
            electrode_positions: list of (row, col) tuples defining valid states
            sigma: spatial decay parameter of the EMF field model
        """

        self.electrode_positions = [tuple(pos) for pos in electrode_positions]
        self.num_states = len(self.electrode_positions)
        self.sigma = sigma
        self.threshold = threshold 

        # each electrode has a preallocated index, len(pos_to_idx) = action_dim 
        self.pos_to_idx = {
            pos: i for i, pos in enumerate(self.electrode_positions)
        }

        # adjacency graph 
        self.neighbor_map = self._build_neighbor_map()

        # initial uniform belief map
        self.belief = np.ones(self.num_states) / self.num_states

    def _build_neighbor_map(self):

        """
        Determining electrode neighbors by connecting each electrode to a neighboring N/E/S/W electrode.

        Returns:
            dict[int, list[int]]: state index → neighbor state indices
        """

        neighbor_map = {}

        # loop through every electrode 
        for i, (r,c) in enumerate(self.electrode_positions):
            neighbors = []

            # legal moves in DMFBs
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                candidate = (r + dr, c + dc)
                if candidate in self.pos_to_idx: # move is valid only if destination electrode exists
                    neighbors.append(self.pos_to_idx[candidate])

            neighbor_map[i] = neighbors

        return neighbor_map
    
    def prediction(self): 

        """
        Chapman Kolmogorov update: 
        Applies the motion model:
            - droplet moves exactly one step per timestep
            - motion is restricted to the electrode graph
            - probability is distributed uniformly among neighbors
            - staying in place is not allowed
        """

        new_belief = np.zeros(self.num_states)

        for i in range(self.num_states):
            p = self.belief[i]
            if p == 0:
                continue

            neighbors = self.neighbor_map[i]

            if len(neighbors) == 0:
                # deadspot, design error, ignore
                continue

            p_per_neighbor = p / len(neighbors)
            for j in neighbors:
                new_belief[j] += p_per_neighbor

        # Normalize to ensure valid probability distribution
        Z = new_belief.sum()
        if Z > 0:
            new_belief /= Z
        else:
            # Fallback: uniform belief if model degenerates
            new_belief[:] = 1.0 / self.num_states

        self.belief = new_belief

    def correction(self, measurement, probe_position, probe_noise=0.03):
        """
        Correction step (Bayesian measurement update).

        Args:
            measurement: EMF reading from the probe
            probe_position: (row, col) tuple of current probe location
            probe_noise: standard deviation of measurement noise
        """

        probe_r, probe_c = probe_position
        likelihood = np.zeros(self.num_states)

        for i, (r, c) in enumerate(self.electrode_positions):
            dist_sq = (probe_r - r)**2 + (probe_c - c)**2
            predicted_emf = np.exp(-dist_sq / (2 * self.sigma**2))

            error_sq = (measurement - predicted_emf)**2
            likelihood[i] = np.exp(-error_sq / (2 * probe_noise**2))

        posterior = self.belief * likelihood
        Z = posterior.sum()

        if Z > 0:
            posterior /= Z
        else:
            # If likelihood collapses, retain prior
            posterior = self.belief.copy()

        self.belief = posterior

    def reset(self):
        """
        Reset belief to uniform distribution over all electrode states.
        """

        self.belief[:] = 1.0 / self.num_states

    def get_belief_distribution(self):
        """
        Get the current belief distribution.
        
        Returns:
            np.ndarray: Current belief probability for each electrode state
        """
        return self.belief.copy()

    def get_action_mask(self):

        """
        Returns a binary action mask of shape (num_states,).
        """

        low_prob = self.threshold*(1/self.num_states)
        mask = self.belief >= low_prob
        mask = mask.astype(np.int32)

        return mask 
    
    def get_state_entropy(self): 

        """
        Computes the entropy of the belief state.

        Consistent low entropy indicates path reduction, without having to directly compute all possible paths. 
        Plotting entropy over time gives insight into how many possible paths remain given the observations so far. High average entropy means many possible paths, low entropy means few possible paths.

        Minimizing entropy is equivalent to maximizing information gain.
        However, for us the rate of change of entropy is deeply important, along with the magnitude of entropy that achieved that rate of change. 
        """

        belief_nonzero = self.belief[self.belief > 1e-12]

        if len(belief_nonzero) == 0:
            return 0.0
        
        entropy = -np.sum(belief_nonzero * np.log2(belief_nonzero))
        H_max = np.log2(self.num_states)
        normalized_entropy = entropy / H_max
        entropy = normalized_entropy 

        return entropy
    
    
    