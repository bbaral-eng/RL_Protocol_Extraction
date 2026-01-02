"""base implementation exists in colab, but is useless as path computations are combinatorially explosive. Protocols longer than ~10 steps will be intractable. """

import numpy as np 
import json 

class path_computations:

    """
    computes all possible droplet paths based on episode observations and uses an exponential decay function to assist in bonus reward calculation 
    """

    def __init__(self, tau, config_path, max_paths = 100_000): 
       
       """
       Args: 
            tau: controls how aggressive the exponential decay is 
            config_path: path to grid configuration json file
            max_paths: maximum number of paths to compute before giving no bonus reward 
       """
       self.tau = tau 
       self.config_path = config_path 
       self.max_paths = max_paths


       self.electrode_positions, self.sigma = self.grid_config(self.config_path)

    def grid_config(self, config_path):

        """ 
        loads grid configuration from json file 
        """

        with open(config_path,'r') as f: 
            grid_configuration = json.load(f)

        electrode_positions = grid_configuration['electrode_positions']
        sigma = grid_configuration['sigma']

        return electrode_positions, sigma
    
    def compute_emf(self, droplet_pos, probe_pos): 

        """
        computing emf values at every electrode position 
        """

        droplet_coords = self.electrode_positions[droplet_pos]
        probe_coords = self.electrode_positions[probe_pos]

        distance = np.sqrt((droplet_coords[0] - probe_coords[0])**2 + (droplet_coords[1] - probe_coords[1])**2)
        emf = np.exp(-distance**2 / (2 * self.sigma**2))

        return emf 

    def neighboring_electrodes(self, position): 

        """
        finds neighboring electrodes for a given position 
        """

        x, y = self.electrode_positions[position]
        neighbors = []

        for idx, (ex, ey) in enumerate(self.electrode_positions):
            if (abs(ex - x) <= 1 and abs(ey - y) <= 1) and (idx != position):
                neighbors.append(idx)

        return neighbors
    
    def path_consistency(self, path, probe_pos): 

        pass 
    
    def exponential_decay(self, n): 

        """
        computes auxiliary loss based on number of paths at time t, based on a saturation function 
        """

        self.aux_loss = np.exp(-(n-1)/self.tau) # n is number of paths computed at the end of sampling. 