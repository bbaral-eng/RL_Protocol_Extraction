# Dummy Actuation Defense

Creates a defense for a DMFB based on dummy electrode actuations that run alongside the real droplet, confusing an attacker's belief without ever touching the true protocol.

## Problem Statement

A defense schedule has multiple objectives: 

- **Obfuscation (attack accuracy)**: dummy actuations should raise attacker's uncertainty of droplet location
- **Number of dummy actuations (count)**: fewer simultaneous actuations are cheaper to run
- **Available routing space (spread)**: dummies should be actuated closer together so there is more available routing space for true protocols. 

The objectives above were chosen by original authors, but can be modified to your liking based on how attacks and defenses evolve. 

## File Breakdown

A genetic algorithm (NSGA-II, via [pymoo](https://pymoo.org)) evolves a population of candidate defense schedules against three objectives simultaneously, and returns pareto front. 

- **`NSGA_II_utils/genome.py`** — encodes a schedule as offsets from the droplet's own position at each timestep
- **`NSGA_II_utils/operators.py`** — mutation and crossover, built to grow and preserve spatially coherent dummy runs rather than scattered noise
- **`NSGA_II_utils/seeding.py`** — seeds part of the initial population with a reservoir-to-reservoir walk instead of pure noise, so evolution isn't discovering good structure from scratch
- **`NSGA_II_utils/rules.py`** — the three fitness objectives
- **`NSGA_II_utils/field.py`** — EMF superposition model (real droplet + dummies)
- **`NSGA_II_utils/problem.py` / `evolve.py`** — wires the above into pymoo and runs the search
- **`NSGA_II_utils/visualize.py`** — Pareto front plots and per-timestep schedule renders

## Usage

### Search from the command line
```bash
python3 run_defense.py                        # random protocol, opens the interactive Pareto front
python3 run_defense.py --protocol r0 c0 r1 c1 ...  # search against a specific protocol
python3 run_defense.py --front                 # reopen the last search's Pareto front
python3 run_defense.py --pick 12                # render the schedule at front index 12
```
Each search saves `last_run.npz` (raw genomes + fitness) and `pareto_front.csv`

### Evaluate against the real attack agent
```bash
jupyter notebook defense.ipynb
```
Loads the pretrained DRQN, searches a front for a fresh protocol, and lets you pick a point off it to test against the agent directly. 

## File Structure

```
defense/
├── run_defense.py              # CLI: search / front / pick
├── defense.ipynb                # end-to-end: agent + protocol + search + evaluation
├── last_run.npz, pareto_front.csv   # most recent search's output
├── NSGA_II_utils/
│   ├── config.py                # grid, EMF, and search hyperparameters
│   ├── genome.py                # genome <-> schedule encoding
│   ├── operators.py             # mutation / crossover
│   ├── seeding.py               # initial population seeding
│   ├── rules.py                 # fitness objectives
│   ├── field.py                 # EMF field model
│   ├── problem.py, evolve.py    # pymoo wiring
│   └── visualize.py             # plotting
└── RL_agent_utils/              # verbatim copies of the attack agent + multipass refinement
```

## Configuration (`NSGA_II_utils/config.py`)
```python
MAX_DUMMIES_PER_STEP = 4     # dummy electrodes allowed to fire at once
MIN_RADIUS = 3.0             # minimum distance a dummy must keep from the real droplet
DUMMY_AMPLITUDE = 0.25       # dummy EMF strength, as a fraction of the real droplet's
POP_SIZE = 1000              # search population size
N_GEN = 50                   # generations per search
SEED_FRACTION = 0.10         # fraction of the initial population that's seeded, not random
```
