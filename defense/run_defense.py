import argparse
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import defense.NSGA_II_utils.config as config
import defense.NSGA_II_utils.evolve as evolve
import defense.NSGA_II_utils.visualize as visualize
from protocol_creation import intelligent_walk
RUN_PATH = 'last_run.npz'
TABLE_PATH = 'pareto_front.csv'

"""
run_defense.py does the following: 
- Two-phase sample run: search for a Pareto front of defenses, inspect it, then render the schedule for picked point.
    python3 run_defense.py             # search, open the front in an interactive 3D window
    python3 run_defense.py --front     # reopen the interactive 3D window from the last search
    python3 run_defense.py --pick 12   # open the timestep-by-timestep plot for point 12
"""

def search():
    protocol = [
        tuple(cell) for cell in
        intelligent_walk(config.OBSTACLES, config.VISIT_POINTS,
                         config.RESERVOIRS, config.GRID_SIZE)
    ]
    print(f"protocol: {len(protocol)} timesteps, {protocol[0]} to {protocol[-1]}")
    print(f"searching (pop {config.POP_SIZE}, {config.N_GEN} generations)...")

    t0 = time.time()
    protocol, X, F = evolve.generate_defense(protocol)
    elapsed = time.time() - t0

    evolve.save_run(RUN_PATH, protocol, X, F)
    np.savetxt(TABLE_PATH, np.column_stack([np.arange(len(F)), -F[:, 0], F[:, 1], F[:, 2]]),
              header='index,count,spread,attack_accuracy', delimiter=',', comments='', fmt='%.6f')

    count, spread, attack_accuracy = -F[:, 0], F[:, 1], F[:, 2]
    print(f"\nsearch took {elapsed:.1f}s")
    print(f"population {config.POP_SIZE} -> {len(F)} non-dominated (Pareto-optimal) out of it")
    print(f"saved {RUN_PATH}, {TABLE_PATH}")
    print(f"\n{'':>16} {'min':>8} {'median':>8} {'max':>8}")
    print(f"{'count':>16} {count.min():>8.3f} {np.median(count):>8.3f} {count.max():>8.3f}")
    print(f"{'spread':>16} {spread.min():>8.3f} {np.median(spread):>8.3f} {spread.max():>8.3f}")
    print(f"{'attack_accuracy':>16} {attack_accuracy.min():>8.3f} {np.median(attack_accuracy):>8.3f} {attack_accuracy.max():>8.3f}")
    print(f"\nfull table in {TABLE_PATH}; run `python run_defense.py --pick <index>` to render one")
    print("opening interactive 3D plot (close the window to exit)...")

    visualize.plot_pareto_front(F, interactive=True)


def front():
    protocol, X, F = evolve.load_run(RUN_PATH)
    print(f"front has {len(F)} points")
    print("opening interactive 3D plot (close the window to exit)...")
    visualize.plot_pareto_front(F, interactive=True)


def render(idx):
    protocol, X, F = evolve.load_run(RUN_PATH)
    if not (0 <= idx < len(F)):
        raise SystemExit(f"index {idx} out of range: front has {len(F)} points")

    schedule, scores = evolve.decode_point(protocol, X, F, idx)

    print(f"point {idx}:")
    print(f"  count            {scores['count']:.3f}")
    print(f"  spread           {scores['spread']:.3f}")
    print(f"  attack_accuracy  {scores['attack_accuracy']:.4f}")

    print("opening the timestep-by-timestep plot (close the window to exit)...")
    visualize.plot_timesteps(protocol, schedule, interactive=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pick', type=int, default=None,
                        help='render the Pareto point at this index from the last search')
    parser.add_argument('--front', action='store_true',
                        help='reopen the interactive 3D plot from the last search, no re-run')
    args = parser.parse_args()

    if args.front:
        front()
    elif args.pick is None:
        search()
    else:
        render(args.pick)

if __name__ == '__main__':
    main()
