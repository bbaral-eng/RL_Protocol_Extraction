import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..')
sys.path.insert(0, os.path.join(_ROOT, 'environment_utils'))
with open(os.path.join(_ROOT, 'configuration', 'grid_configuration.json'), 'r') as f:
    _GRID = json.load(f)

GRID_SIZE = _GRID['gridSize']
REAL_SIGMA = _GRID['sigma']
OBSTACLES = [tuple(o) for o in _GRID['obstacles']]
RESERVOIRS = [tuple(r) for r in _GRID['reservoir_positions']]
ELECTRODES = [tuple(e) for e in  _GRID['electrode_positions']]
VISIT_POINTS = [tuple(v) for v in _GRID['visit_points']]
DUMMY_AMPLITUDE = 0.25             # fraction of the EMF a real actuation emits
DUMMY_SIGMA = REAL_SIGMA * 0.9     # fraction of the spread of a real actuation

MAX_DUMMIES_PER_STEP = 4            # n maximum dummy firings per timestep that algorithm can do
MIN_RADIUS = 3.0                    # hard floor so a dummy's field doesn't merge into and brighten the droplet's own peak
POP_SIZE = 1000
N_GEN = 50
SEED = 1

KINEMATIC_SCALE = 1.0                # decay rate for movement beyond what elapsed time could physically explain
SPLIT_JITTER = 1.5                   # how far a Path Split sibling can land from the dummy it copied
SEED_FRACTION = 0.20                 # fraction of initial population built by seeding.py instead of pure noise; 0 disables it

USE_CAPTURE_ZONE_SPREAD = True      # objective 2: False = convex-hull spread_score, True = capture_zone_score
