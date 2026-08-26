import random

import networkx as nx
import numpy as np
from pymoo.core.sampling import Sampling
import defense.NSGA_II_utils.config as config
import defense.NSGA_II_utils.genome as genome

# seeding.py: builds the initial population. A fraction (config.SEED_FRACTION) starts as hand-made genomes instead of pure noise, so evolution doesn't have to discover a good
# shape from scratch. Swap `seed_fn` for your own (protocol, variant) -> genome_row function to try a different seed shape.

# default is a 2x2 dummy electrode actuation that resembles a protocol. 

_OBSTACLES = set(config.OBSTACLES)
_ORTHOGONAL = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
_CORNER = {('up', 'left'): (-1, -1), ('up', 'right'): (-1, 1), ('down', 'left'): (1, -1), ('down', 'right'): (1, 1),}
_ALL_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _is_valid(cell, true_cell):
    """
    determines if electrode cell is usable for a dummy: on the grid, not an obstacle, far enough from the droplet.
    """
    r, c = cell
    row, col = true_cell

    on_grid = 0 <= r < config.GRID_SIZE and 0 <= c < config.GRID_SIZE
    if not on_grid:
        return False

    is_obstacle = cell in _OBSTACLES
    if is_obstacle:
        return False

    far_enough = np.hypot(r - row, c - col) >= config.MIN_RADIUS
    return far_enough


def _distance_map(graph, target):
    """
    Shortest number of grid steps from every reachable cell to a specific target
    """
    return nx.shortest_path_length(graph, target=target)


def _next_step_toward(current, target, dist_from_target):
    """
    One neighbor that's a step closer to target, per dist_from_target. Ties (more than one neighbor equally close) are broken by moving along whichever axis (row or column) 
    currently has the bigger remaining gap.
    """
    if current == target:
        return current

    current_dist = dist_from_target[current]
    row_gap = target[0] - current[0]
    col_gap = target[1] - current[1]
    prefer_row_move = abs(row_gap) >= abs(col_gap)

    candidates = []
    for d_row, d_col in _ORTHOGONAL.values():
        neighbor = (current[0] + d_row, current[1] + d_col)
        if dist_from_target.get(neighbor) == current_dist - 1:
            moved_along_row = d_row != 0
            candidates.append((neighbor, moved_along_row))

    if not candidates:
        return current

    candidates.sort(key=lambda pair: pair[1] != prefer_row_move)
    return candidates[0][0]

def _walk_reservoirs(static_graph, protocol, reservoir_order):
    """
    Greedy per-timestep walk through reservoir_order (randomly selected). Each step tries to move one cell closer to the current target; if the real droplet's MIN_RADIUS zone blocks that move,
    stays in place; if even staying is unsafe, that timestep is left inactive. Advances to the next reservoir once the current one is reached; holds position once the list runs
    out. Returns one cell (or None for "inactive") per timestep.
    """
    total_steps = len(protocol)
    positions = [None] * total_steps

    start_cell = next((r for r in reservoir_order if _is_valid(r, protocol[0])), None)
    if start_cell is None:
        return positions

    remaining_targets = [r for r in reservoir_order if r != start_cell]
    current = start_cell
    dist_to_target = _distance_map(static_graph, remaining_targets[0]) if remaining_targets else None

    for t in range(total_steps):
        true_cell = protocol[t]

        if remaining_targets and current == remaining_targets[0]:
            remaining_targets.pop(0)
            dist_to_target = _distance_map(static_graph, remaining_targets[0]) if remaining_targets else None

        if remaining_targets:
            step = _next_step_toward(current, remaining_targets[0], dist_to_target)
        else:
            step = current

        if _is_valid(step, true_cell):
            current = step
            positions[t] = current
        elif _is_valid(current, true_cell):
            positions[t] = current
        else:
            positions[t] = None

    return positions


def _pack_cluster(base_cell, true_cell):
    """
    Complete a 2x2 square around the base cell if the neighbors allow it; otherwise take whichever nearby cells are valid, no shape required.
    """
    base_row, base_col = base_cell

    def neighbor(direction):
        d_row, d_col = _ORTHOGONAL[direction]
        return (base_row + d_row, base_col + d_col)

    valid_horizontals = [d for d in ('left', 'right') if _is_valid(neighbor(d), true_cell)]
    valid_verticals = [d for d in ('down', 'up') if _is_valid(neighbor(d), true_cell)]

    if valid_horizontals and valid_verticals:
        h_dir, v_dir = valid_horizontals[0], valid_verticals[0]
        d_row, d_col = _CORNER[(v_dir, h_dir)]
        corner_cell = (base_row + d_row, base_col + d_col)
        if _is_valid(corner_cell, true_cell):
            return [base_cell, neighbor(h_dir), neighbor(v_dir), corner_cell]

    fallback_cells = []
    for d_row, d_col in _ALL_NEIGHBORS:
        cell = (base_row + d_row, base_col + d_col)
        if _is_valid(cell, true_cell):
            fallback_cells.append(cell)
        if len(fallback_cells) == config.MAX_DUMMIES_PER_STEP - 1:
            break

    return [base_cell] + fallback_cells


def clustered_seed(protocol, variant=0):
    """
    Build one seed genome: a reservoir-to-reservoir base walk (obstacles avoided, MIN_RADIUS from the real droplet checked live every step) plus a packed dummy cluster around it at
    every timestep it's active.
    """
    static_graph = nx.grid_2d_graph(config.GRID_SIZE, config.GRID_SIZE)
    static_graph.remove_nodes_from(_OBSTACLES)

    real_start, real_end = protocol[0], protocol[-1]
    candidate_reservoirs = [r for r in config.RESERVOIRS if r != real_start and r != real_end]
    reservoir_order = candidate_reservoirs[:]
    random.Random(variant).shuffle(reservoir_order)

    positions = _walk_reservoirs(static_graph, protocol, reservoir_order)

    genome_row = np.zeros(genome.n_genes(len(protocol)))
    for t, base_cell in enumerate(positions):
        if base_cell is None:
            continue
        true_cell = protocol[t]
        cluster = _pack_cluster(base_cell, true_cell)

        for s, (r, c) in enumerate(cluster[:config.MAX_DUMMIES_PER_STEP]):
            i = genome.electrode_start(t, s)
            genome_row[i] = 1.0
            genome_row[i + 1] = r - true_cell[0]
            genome_row[i + 2] = c - true_cell[1]

    return genome_row


class DefenseSampling(Sampling):
    """
    Fills seed_fraction of the initial population with seed_fn(protocol, variant), the rest with plain uniform random sampling within bounds (pymoo's usual default).
    """

    def __init__(self, seed_fn=clustered_seed, seed_fraction=None):
        super().__init__()
        self.seed_fn = seed_fn
        self.seed_fraction = config.SEED_FRACTION if seed_fraction is None else seed_fraction

    def _do(self, problem, n_samples, **kwargs):
        rng = np.random.default_rng()
        n_seed = round(self.seed_fraction * n_samples)
        X = np.empty((n_samples, problem.n_var))

        for k in range(n_seed):
            X[k] = self.seed_fn(problem.protocol, variant=k)
        for k in range(n_seed, n_samples):
            X[k] = rng.uniform(problem.xl, problem.xu)

        return X
