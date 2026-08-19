import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import defense.NSGA_II_utils.config as config
import defense.NSGA_II_utils.genome as genome
from defense.NSGA_II_utils.operators import DefenseCrossover, DefenseMutation
from defense.NSGA_II_utils.problem import DefenseProblem


# evolve.py simply glues together all components for the fenetic evol algoritm to work, and actually runs NSGA-II 

def generate_defense(protocol, pop_size=None, n_gen=None, seed=None, verbose=False):
    """
    Take a protocol and search for the Pareto front of defenses for it. 
    - Returns (protocol, X, F), every point on the front, undecoded.
    """
    protocol = [tuple(cell) for cell in protocol]

    result = minimize(
        DefenseProblem(protocol),
        NSGA2(pop_size=pop_size or config.POP_SIZE, mutation=DefenseMutation(), crossover=DefenseCrossover()), ('n_gen', n_gen or config.N_GEN), seed=config.SEED if seed is None else seed, verbose=verbose,)
    F = np.atleast_2d(result.F)
    X = np.atleast_2d(result.X)
    return protocol, X, F


def decode_point(protocol, X, F, idx):
    """Decode Pareto point `idx` into a schedule and its scores."""
    schedule = genome.decode(X[idx], protocol)
    scores = {
        'count': float(-F[idx][0]),        # pre-negated in rules.py, undo for display
        'spread': float(F[idx][1]),        # already [0,1], 0=best, no negation
        'attack_accuracy': float(F[idx][2]),  # already [0,1], 0=best, no negation
    }
    return schedule, scores


def save_run(path, protocol, X, F):
    np.savez(path, protocol=np.array(protocol), X=X, F=F)


def load_run(path):
    data = np.load(path)
    protocol = [tuple(int(v) for v in row) for row in data['protocol']]
    return protocol, data['X'], data['F']
