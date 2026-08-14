# importing necessary libraries 
import pygame 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


"_____________________________________________________________rendering functions for enviroment_____________________________________________________________"

"Rendering functions for the environment, including the voltage grid, obstacles, droplets, and agents. The rendering is done using Pygame for real-time visualization."

def initialize_render(render_mode, window, window_size): 
    "Initialize the Pygame window for rendering the environment."
    if window is None and render_mode == 'human':
        pygame.init()
        pygame.display.init()
        window = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Voltage Distribution RL Environment")
        clock = pygame.time.Clock()
        return window, clock
    return window, None

def render_env(voltage_grid, agent_location, droplet_location, obstacles, window, clock, render_mode, gridSize, window_size, render_fps=8):
    "rendering actual environment w/ voltage grid, obstacles, droplets, and agents"
   
    canvas = pygame.Surface((window_size, window_size))
    canvas.fill((255, 255, 255))  
    pix_square_size = window_size / gridSize
    obstacle_set = set(tuple(obstacle) for obstacle in obstacles)
    colormap = cm.get_cmap('inferno')

    for row in range(gridSize):
        for col in range(gridSize):
            rect = pygame.Rect(col * pix_square_size, row * pix_square_size,
                            pix_square_size, pix_square_size)
            
            if (row, col) in obstacle_set:
                pygame.draw.rect(canvas, (255, 255, 255), rect)
            else:
                pygame.draw.rect(canvas, (0, 0, 0), rect)         
                pygame.draw.rect(canvas, (255, 255, 255), rect, 1)
            
            voltage_value = np.clip(voltage_grid[row, col] / 1, 0, 1)
            
            if voltage_value > 0.1:
                color_rgb = (np.array(colormap(voltage_value)[:3]) * 255).astype(int)
                pygame.draw.rect(canvas, color_rgb, rect)
                
                if (row, col) not in obstacle_set:
                    pygame.draw.rect(canvas, (255, 255, 255), rect, 1)

    # Draw droplets
    droplet_center = (droplet_location[::-1] + 0.5) * pix_square_size
    pygame.draw.circle(canvas, (0, 0, 255), tuple(droplet_center.astype(int)), int(pix_square_size / 3))

    # Draw agent 
    agent_center = (agent_location[::-1] + 0.5) * pix_square_size
    pygame.draw.circle(canvas, (255, 0, 0), agent_center.astype(int), int(pix_square_size / 3))

    # handling render modes
    if render_mode == "human" and window is not None:
        window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        if clock:
            clock.tick(render_fps)  # Use configurable FPS
        return None  
    else:
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

def cleanup_render():
    "Clean up pygame resources"
    try:
        pygame.display.quit()
        pygame.quit()
    except:
        pass  # Ignore errors if pygame wasn't initialized