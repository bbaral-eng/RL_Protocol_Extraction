# importing necessary libraries 
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np 

"_____________________________________________________________Configurations for environment_____________________________________________________________"

"define the grid size, number of droplets, and number of agents (set to 1 and can only be one). The grid will be a square matrix of gridSize (n x n). The number of droplets and agents can be set as per requirement."

# defining global variables (editable)
gridSize = 11               # grid size 
numDroplets = 1             # number of droplets 
numAgents = 1               # number of agents (capped at 1)
sigma = 1.5                 # standard deviation for Gaussian noise in voltage signal from electrode actuations 
window_size = 600           # window size for rendering 
render_fps = 8              # render fps 
k = 2                       # adjustable depth of voltage_history, higher k means less compute time but less memory, lower k means more compute time but more memory
N = gridSize//k             # hyperparameter for tensor depth in voltage_history 
threshold = 0.5             # belief state threshold for action masking: threshold*(1/num_states), states with belief below this value will be masked out

"______________once the grid is set, do not change the size of the grid. Run the code to define obstacles, reservoirs, and visit points.________________"

"It is advised to draw reservoirs as singular locations in the general area where you want to place droplets."

#______________________________________________ do not edit code, used for remaining grid constraints ____________________________________________#
def grid_editor(gridSize): 
    grid = np.ones((gridSize, gridSize))

    obstacles = set() # set of obstacles
    reservoir_positions = set() # set of reservoir positions
    visit_points = set() # set of visited points by droplet 

    # Drawing mode
    mode = {'current': 'obstacle'}
    drawing = {'active': False}
  
    obstacle_color = 'black'
    reservoir_color = 'blue'
    visit_point_color = 'green'

    # visualize grid 
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120) 
    ax.set_xlim(0, gridSize)
    ax.set_ylim(gridSize, 0) # invert y-axis to have (0,0) at the top left corner
    ax.grid(True)
    ax.set_xticks(np.arange(0, gridSize, 1))
    ax.set_yticks(np.arange(0, gridSize, 1))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Click and drag to add obstacles, reservoirs, and visit points.")

    ax.tick_params(axis='both', 
                   which='both', 
                   bottom=False, 
                   left=False, 
                   labelbottom=False, 
                   labelleft=False)
    
    legend_elements = [
        Patch(facecolor=obstacle_color, edgecolor='black', label='Obstacles'),
        Patch(facecolor=reservoir_color, edgecolor='black', label='Reservoirs'),
        Patch(facecolor=visit_point_color, edgecolor='black', label='Visit Points')
    ]

    # visualize legend and make it interactive 
    legend = ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    for legend_patch in legend.get_patches():
        legend_patch.set_picker(True)

    # Function to update the plot with a new square
    def update_plot(x, y, color):
        square = plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='gray')
        ax.add_patch(square)
        fig.canvas.draw_idle()

    # Function to add point based on current mode
    def add_point(x, y):
        if mode['current'] == 'obstacle':
            if (x, y) not in obstacles:
                obstacles.add((x, y))
                update_plot(x, y, obstacle_color)

        elif mode['current'] == 'reservoir':
            if (x, y) not in reservoir_positions:
                reservoir_positions.add((x, y))
                update_plot(x, y, reservoir_color)

        elif mode['current'] == 'visit':
            if (x, y) not in visit_points:
                visit_points.add((x, y))
                update_plot(x, y, visit_point_color)

    # Function to handle mouse button press
    def on_press(event):
        if event.inaxes == ax:
            drawing['active'] = True
            x, y = int(event.xdata), int(event.ydata)
            add_point(x, y)

    # Function to handle mouse button release
    def on_release(event):
        drawing['active'] = False

    # Function to handle mouse motion
    def on_motion(event):
        if drawing['active'] and event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            add_point(x, y)

    def on_pick(event):
        label = event.artist.get_label()
        if label == 'Obstacles':
            mode['current'] = 'obstacle'
            ax.set_title("Mode: Obstacles (Click and drag)")
        elif label == 'Reservoirs':
            mode['current'] = 'reservoir'
            ax.set_title("Mode: Reservoirs (Click and drag)")
        elif label == 'Visit Points':
            mode['current'] = 'visit'
            ax.set_title("Mode: Visit Points (Click and drag)")
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('pick_event', on_pick)

    plt.show()
    
    return tuple(obstacles), tuple(reservoir_positions), tuple(visit_points)

def main():
    """Main function for grid configuration"""
    obstacles, reservoir_positions, visit_points = grid_editor(gridSize)

    # inverse x and y coordinates for consistency with RL Environment 
    obstacles = [(y, x) for (x, y) in obstacles]
    reservoir_positions = [(y, x) for (x, y) in reservoir_positions]
    visit_points = [(y, x) for (x, y) in visit_points]

    # identifying all electrode positions in the grid 
    electrode_positions = [
        (i, j)
        for i in range(gridSize)
        for j in range(gridSize)
        if (i, j) not in obstacles
    ]

    # Save the grid configuation as a json file 
    import json
    import os 

    grid_data = {
        'render_fps': render_fps,
        'window_size': window_size,
        'gridSize': gridSize,
        'numDroplets': numDroplets,
        'numAgents': numAgents,
        'sigma': sigma,
        'obstacles': obstacles,
        'reservoir_positions': reservoir_positions,
        'visit_points': visit_points,
        'electrode_positions': electrode_positions,
        'k': k,
        'N': N,
        'threshold': threshold
    }

    current_folder = os.path.dirname(os.path.abspath(__file__))
    json_filename = "grid_configuration.json"
    json_path = os.path.join(current_folder, json_filename)

    with open(json_path, 'w') as f:
        json.dump(grid_data, f, indent=2)

    print(f"data saved to {json_path}")

if __name__ == "__main__":
    main()