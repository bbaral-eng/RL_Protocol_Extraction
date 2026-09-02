import os
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
os.chdir(os.path.join(_REPO, "defense"))                     
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "environment_utils"))

from defense.NSGA_II_utils import config
from defense.NSGA_II_utils import evolve
from protocol_creation import intelligent_walk

OUT = os.path.join(_HERE, "multi_protocol_defense.npz")

"""
Creates N random droplet protocols and evolves a Pareto front of dummy-actuation defenses for each one, then saves everything to generated_defense_data.npz.

  generate()                      # 10 protocols, NSGA defaults from config.py
  generate(n_protocols=25)        # more protocols
  generate(pop_size=400, n_gen=30)  # faster / rougher search

plot_generated_data.py reads the .npz, picks one defense per protocol, and runs the attacker.
"""


def generate(n_protocols=10, pop_size=config.POP_SIZE, n_gen=config.N_GEN, seed0=0, out=OUT):
    """
    Make n_protocols random protocols, evolve a defense front for each, save to `out`.
    """

    protocols, fronts_X, fronts_F = [], [], []

    for k in range(n_protocols):
        random.seed(seed0 + k)                              
        protocol = [tuple(c) for c in intelligent_walk(
            config.OBSTACLES, config.VISIT_POINTS, config.RESERVOIRS, config.GRID_SIZE)]

        _, X, F = evolve.generate_defense(protocol, pop_size=pop_size, n_gen=n_gen, seed=1)

        protocols.append(np.array(protocol))
        fronts_X.append(X)                                   
        fronts_F.append(F)                                   
        print(f"[{k + 1}/{n_protocols}] protocol length {len(protocol)}, "
              f"front {len(F)} points, lowest attack_accuracy {F[:, 2].min():.3f}")

    def _obj_array(list_of_arrays):
        a = np.empty(len(list_of_arrays), dtype=object)   
        for i, item in enumerate(list_of_arrays):
            a[i] = item
        return a

    np.savez(
        out,
        protocols=_obj_array(protocols),
        X=_obj_array(fronts_X),
        F=_obj_array(fronts_F),
        pop_size=pop_size, n_gen=n_gen, n_protocols=n_protocols, seed0=seed0,
    )
    print(f"\nsaved {out}")
    return out


if __name__ == "__main__":
    generate()
