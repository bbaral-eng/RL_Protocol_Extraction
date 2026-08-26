import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
import defense.NSGA_II_utils.config as config
import defense.NSGA_II_utils.genome as genome


# operators.py: mutation and crossover for the defense genome.
_NEAR_STEPS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

def _slot_active(row, t, s):
    """
    returns if slot s is actuated at timestep t 
    """
    return row[genome.electrode_start(t, s)] >= 0.5

def _turn_on(row, i, dr, dc, problem):
    row[i] = 1.0
    row[i + 1] = np.clip(dr, problem.xl[i + 1], problem.xu[i + 1])
    row[i + 2] = np.clip(dc, problem.xl[i + 2], problem.xu[i + 2])

def _turn_off(row, i):
    row[i] = 0.0


class DefenseMutation(Mutation):
    """
    One random structural edit per individual per generation, picked uniformly from the 4 options below and avoids one edit undoing another in the same pass.
    """

    def __init__(self, **kwargs):
        super().__init__(prob=1.0, **kwargs)

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        if random_state is None:  # older pymoo builds don't always thread this through
            random_state = np.random.default_rng()

        X = X.copy()
        n_steps = len(problem.protocol)
        n_slots = config.MAX_DUMMIES_PER_STEP
        ops = [self._toggle, self._extend, self._relocate, self._cluster]

        for row in X:
            op = ops[int(random_state.integers(len(ops)))]
            op(row, n_steps, n_slots, problem, random_state)

        return X

    def _toggle(self, row, n_steps, n_slots, problem, rs):
        """
        Flip one random cell: birth (fresh random position) if off, prune if on.
        """
        t = int(rs.integers(n_steps))
        s = int(rs.integers(n_slots))
        i = genome.electrode_start(t, s)

        if _slot_active(row, t, s):
            _turn_off(row, i)
        else:
            dr = rs.uniform(problem.xl[i + 1], problem.xu[i + 1])
            dc = rs.uniform(problem.xl[i + 2], problem.xu[i + 2])
            _turn_on(row, i, dr, dc, problem)

    def _extend(self, row, n_steps, n_slots, problem, rs):
        """
        Grow a run by one timestep, moving no more than a real droplet could (1 cell).
        """
        s = int(rs.integers(n_slots))
        active = [t for t in range(n_steps) if _slot_active(row, t, s)]
        if not active:
            return

        t0 = int(rs.choice(active))
        neighbors = [t for t in (t0 - 1, t0 + 1) if 0 <= t < n_steps and not _slot_active(row, t, s)]
        if not neighbors:
            return
        t_new = int(rs.choice(neighbors))

        i_ref, i_new = genome.electrode_start(t0, s), genome.electrode_start(t_new, s)
        abs_ref = np.array(problem.protocol[t0]) + row[i_ref + 1:i_ref + 3]
        d_row, d_col = _NEAR_STEPS[int(rs.integers(len(_NEAR_STEPS)))]
        abs_new = abs_ref + np.array([d_row, d_col])

        offset_new = abs_new - np.array(problem.protocol[t_new])
        _turn_on(row, i_new, offset_new[0], offset_new[1], problem)

    def _relocate(self, row, n_steps, n_slots, problem, rs):
        """
        Slide a whole run to a new spot, shape unchanged.
        """
        s = int(rs.integers(n_slots))
        active = [t for t in range(n_steps) if _slot_active(row, t, s)]
        if not active:
            return

        span = float(config.GRID_SIZE - 1)
        d_dr = rs.uniform(-span / 4, span / 4)
        d_dc = rs.uniform(-span / 4, span / 4)

        for t in active:
            i = genome.electrode_start(t, s)
            row[i + 1] = np.clip(row[i + 1] + d_dr, problem.xl[i + 1], problem.xu[i + 1])
            row[i + 2] = np.clip(row[i + 2] + d_dc, problem.xl[i + 2], problem.xu[i + 2])

    def _cluster(self, row, n_steps, n_slots, problem, rs):
        """
        Copy an active dummy into an empty slot at the same timestep, placed close by.
        """
        if n_slots < 2:
            return

        t = int(rs.integers(n_steps))
        active = [s for s in range(n_slots) if _slot_active(row, t, s)]
        empty = [s for s in range(n_slots) if not _slot_active(row, t, s)]
        if not active or not empty:
            return

        src = genome.electrode_start(t, int(rs.choice(active)))
        dst = genome.electrode_start(t, int(rs.choice(empty)))
        jitter = config.SPLIT_JITTER

        dr = row[src + 1] + rs.uniform(-jitter, jitter)
        dc = row[src + 2] + rs.uniform(-jitter, jitter)
        _turn_on(row, dst, dr, dc, problem)


class DefenseCrossover(Crossover):
    def __init__(self, **kwargs):
        super().__init__(2, 2, **kwargs)

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        if random_state is None:  # older pymoo builds don't always thread this through
            random_state = np.random.default_rng()

        _, n_matings, n_var = X.shape
        n_steps = len(problem.protocol)
        n_slots = config.MAX_DUMMIES_PER_STEP
        Q = np.empty((2, n_matings, n_var))

        for m in range(n_matings):
            a, b = X[0, m], X[1, m]
            c1, c2 = self._run_safe_swap(a, b, n_steps, n_slots, random_state)
            Q[0, m], Q[1, m] = c1, c2

        return Q

    def _run_safe_swap(self, a, b, n_steps, n_slots, rs):
        """
        Per-slot time-slice swap. Only cuts where the child's own boundary (a before the cut, b after) wouldn't glue two unrelated runs together.
        """
        c1, c2 = a.copy(), b.copy()

        for s in range(n_slots):
            safe_cuts = [
                t for t in range(1, n_steps)
                if not (_slot_active(a, t - 1, s) and _slot_active(b, t, s))
                and not (_slot_active(b, t - 1, s) and _slot_active(a, t, s))
            ]
            if not safe_cuts:
                continue

            cut = int(rs.choice(safe_cuts))
            for t in range(cut, n_steps):
                i = genome.electrode_start(t, s)
                c1[i:i + 3], c2[i:i + 3] = b[i:i + 3], a[i:i + 3]

        return c1, c2
