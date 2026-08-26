import numpy as np
import defense.NSGA_II_utils.config as config

GENES_PER_SLOT = 3   
# GENES_PER_SLOT = (active, dr, dc)
# active: [0,1] decides if electrode actuated or not
# dr, dc: row/col offset from droplet
# one slot contains these three genes above

_OBSTACLES = set(config.OBSTACLES)

# genome.py defines candidate defense solutions. 
# a genome is up to MAX_DUMMIES_PER_STEP*(active, dr, dc) from t = 0 to t = t_end. 
# for a timesteps up to 30, you can have a genome that is MAX_DUMMIES_PER_STEP(max number of slots)*GENES_PER_SLOT*30 in length. 

def electrode_start(t, electrode_index):
    """
    given timestep and electrode index, returns where the electrode's slot of genes begin in the flat genome vector. 
    """
    return (t * config.MAX_DUMMIES_PER_STEP + electrode_index) * GENES_PER_SLOT


def bounds(protocol):
    """
    Lower and upper bound for every slot: active is [0,1], while dr, dc are with respect to the grid size, and a droplet's location @ a timestep. 
    """
    lower_rows, upper_rows = [], []

    for row, col in protocol:
        dr_lo, dr_hi = -row, config.GRID_SIZE - 1 - row
        dc_lo, dc_hi = -col, config.GRID_SIZE - 1 - col
        for _ in range(config.MAX_DUMMIES_PER_STEP):
            lower_rows.append([0.0, dr_lo, dc_lo])
            upper_rows.append([1.0, dr_hi, dc_hi])

    lower = np.array(lower_rows, dtype=float).ravel()
    upper = np.array(upper_rows, dtype=float).ravel()
    return lower, upper


def n_genes(n_steps):
    """
    returns max length of genome. 
    """
    return n_steps * config.MAX_DUMMIES_PER_STEP * GENES_PER_SLOT


def decode(x, protocol):
    """
    input: x (genome), protocol
    Turn a genome into a schedule: one list of dummy electrode actuations per timestep.
    Each active electrode is a (row, col) offset from the droplet's own position
    for that timestep, active decides if electrode fires, dr/dc decide where.
    """
    schedule = []
    i = 0

    for row, col in protocol:
        cells = []

        for _ in range(config.MAX_DUMMIES_PER_STEP):
            active, dr, dc = x[i:i + GENES_PER_SLOT]
            i += GENES_PER_SLOT

            if active < 0.5:
                continue

            r = int(round(row + dr))
            c = int(round(col + dc))

            if not (0 <= r < config.GRID_SIZE and 0 <= c < config.GRID_SIZE):
                continue
            if (r, c) in _OBSTACLES:
                continue
            if np.hypot(r - row, c - col) < config.MIN_RADIUS:
                continue
            if (r, c) in cells:
                continue

            cells.append((r, c))

        schedule.append(cells)

    # actual defense schedule 
    return schedule
