# importing necessary libraries 
import itertools
import random 
import networkx as nx    
import numpy as np 

"____________________________________________ choosing droplet path from reservoir to visit points to reservoir ____________________________________________"

"intelligent_walk allows you to compute a droplet path from point to point based on the grid layout and design chosen in config.py"

def intelligent_walk(obstacles, visit_points, reservoir_positions, gridSize, max_paths=100_000): 

    "Predefined walk for the grid based on intelligent pathfinding"

    # list to tuple conversion for networkx 
    reservoir_positions = tuple(tuple(reservoir_position) for reservoir_position in reservoir_positions)
    visit_points = tuple(tuple(visit_points) for visit_points in visit_points)
    obstacles = tuple(tuple(obstacle) for obstacle in obstacles)

    # implementing Djikstra's Algorithm using nx
    Graph = nx.grid_2d_graph(gridSize, gridSize)

    # Remove obstacles from the graph
    Graph.remove_nodes_from(obstacles)

    # Randomly select start/end reservoirs and 4 unique visit points
    start_pos, end_pos = random.sample(reservoir_positions, 2)
    
    # Select 4 unique visit points (or all available if less than 4)
    num_visit_points = min(4, len(visit_points))
    selected_visit_points = random.sample(list(visit_points), num_visit_points)

    try:
        # Build path through all visit points: start -> vp1 -> vp2 -> vp3 -> vp4 -> end
        random_path = []
        current_pos = start_pos
        
        for i, visiting_point in enumerate(selected_visit_points):
            # Get path from current position to next visit point
            all_paths = list(itertools.islice(
                nx.all_shortest_paths(Graph, source=current_pos, target=visiting_point), 
                max_paths
            ))
            if not all_paths:
                raise nx.NetworkXNoPath(f"No path from {current_pos} to {visiting_point}")
            path_segment = random.choice(all_paths)
            
            # Append path segment (skip first node if not first segment to avoid duplicates)
            if i == 0:
                random_path.extend(path_segment)
            else:
                random_path.extend(path_segment[1:])
            
            current_pos = visiting_point
        
        # Get path from last visit point to end
        all_paths_end = list(itertools.islice(
            nx.all_shortest_paths(Graph, source=current_pos, target=end_pos), 
            max_paths
        ))
        if not all_paths_end:
            raise nx.NetworkXNoPath(f"No path from {current_pos} to {end_pos}")
        path_end = random.choice(all_paths_end)
        
        # Append final segment 
        random_path.extend(path_end[1:])
        
    except nx.NetworkXNoPath as e:
        print(f"Warning: {e}")
        print("Falling back to direct path between reservoirs")
        # Fallback: direct path between reservoirs
        try:
            random_path = nx.shortest_path(Graph, source=start_pos, target=end_pos)
        except nx.NetworkXNoPath:
            print("Error: No valid path exists in the grid!")
            # Last resort: return just the start and end positions
            random_path = [start_pos, end_pos]

    path_coords = np.array(random_path)
    length = len(path_coords)

    return path_coords