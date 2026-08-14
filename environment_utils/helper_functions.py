# importing necessary libraries 
import numpy as np


"_____________________________________________________________ helper functions which are used in the main file _____________________________________________________________"

"voltage_grid defines voltage across the entire grid as electrodes get actuated. get_obs records observations as the agent (probe) moves from electrode to electrode."

def voltage_grid(droplet_location, sigma, obstacles, gridSize):  # creating the voltage grid 

    # Initialize the voltage grid with zeros
    voltage_values = np.zeros((gridSize, gridSize))  

    # Iterate over each droplet location
    for droplet_location in [droplet_location]:
        chosenRow, chosenCol = droplet_location

        x = np.arange(gridSize)  
        y = np.arange(gridSize)  
        xv, yv = np.meshgrid(x, y)
            
        # Creating a grid of distances from the droplets to all other grid points
        distances = np.sqrt((xv - chosenRow)**2 + (yv - chosenCol)**2)

        # Calculating voltage values using a Gaussian distribution
        values = np.exp(-distances**2 / (2 * sigma**2))

        # Combine the voltage values from all droplets
        voltage_values += values.T

        # round voltage to 3 decimal places 
        voltage_values = np.round(voltage_values, 3)

        # set voltage values to 0 if less than 0.001
        voltage_values[voltage_values < 0.001] = 0

    return voltage_values

def get_obs(voltage_grid, agent_location, _droplet_path_index, voltage_history, length):  # returns the observation of the agent
        voltage = voltage_grid[agent_location[0], agent_location[1]]
        probe_position = agent_location.astype(np.int32)
        time_step = np.array([_droplet_path_index], dtype=np.float32)/(length-2) # -2 because 1) we index at 0, 2) first "step" DNE since its technically in reset
        voltage_history = voltage_history.astype(np.float32)
        
        return {
            'voltage': np.array([voltage], dtype=np.float32),
            'probe position': probe_position,
            'time step': time_step,
            'voltage history': voltage_history
        }