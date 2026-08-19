import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
import defense.NSGA_II_utils.config as config
import defense.NSGA_II_utils.genome as genome


class DefenseMutation(Mutation):

    def __init__(self, flip_prob=0.05, shift_prob=0.15, split_prob=0.1, **kwargs):
        super().__init__(prob=1.0, **kwargs)
        self.flip_prob = flip_prob
        self.shift_prob = shift_prob
        self.split_prob = split_prob

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        if random_state is None:  # older pymoo builds don't always thread this through
            random_state = np.random.default_rng()

        X = X.copy()
        n_steps = len(problem.protocol)
        n_slots = config.MAX_DUMMIES_PER_STEP

        for row in X:
            self._random_flip(row, n_steps, n_slots, problem, random_state)
            self._cluster_shift(row, n_steps, n_slots, problem, random_state)
            self._path_split(row, n_steps, n_slots, problem, random_state)

        return X

    def _random_flip(self, row, n_steps, n_slots, problem, rs):
        """Turn a single electrode on or off."""
        if rs.random() >= self.flip_prob:
            return

        i = genome.electrode_start(int(rs.integers(n_steps)), int(rs.integers(n_slots)))
        self._set_active(row, i, row[i] < 0.5, problem, rs)

    def _cluster_shift(self, row, n_steps, n_slots, problem, rs):
        """Slide one slot's whole run of active timesteps by the same offset, keeping its shape intact, just moved somewhere else."""
        if rs.random() >= self.shift_prob:
            return

        s = int(rs.integers(n_slots))
        active_steps = [t for t in range(n_steps) if row[genome.electrode_start(t, s)] >= 0.5]
        if not active_steps:
            return

        span = float(config.GRID_SIZE - 1)
        d_dr = rs.uniform(-span / 4, span / 4)
        d_dc = rs.uniform(-span / 4, span / 4)

        for t in active_steps:
            i = genome.electrode_start(t, s)
            row[i + 1] = np.clip(row[i + 1] + d_dr, problem.xl[i + 1], problem.xu[i + 1])
            row[i + 2] = np.clip(row[i + 2] + d_dc, problem.xl[i + 2], problem.xu[i + 2])

    def _path_split(self, row, n_steps, n_slots, problem, rs):
        """Copy an active dummy into a currently-empty slot at the same timestep, but moved a bit"""
        if rs.random() >= self.split_prob or n_slots < 2:
            return

        t = int(rs.integers(n_steps))
        active = [s for s in range(n_slots) if row[genome.electrode_start(t, s)] >= 0.5]
        empty = [s for s in range(n_slots) if row[genome.electrode_start(t, s)] < 0.5]
        if not active or not empty:
            return

        src = genome.electrode_start(t, int(rs.choice(active)))
        dst = genome.electrode_start(t, int(rs.choice(empty)))
        jitter = config.SPLIT_JITTER

        row[dst] = 1.0
        row[dst + 1] = np.clip(row[src + 1] + rs.uniform(-jitter, jitter),
                               problem.xl[dst + 1], problem.xu[dst + 1])
        row[dst + 2] = np.clip(row[src + 2] + rs.uniform(-jitter, jitter),
                               problem.xl[dst + 2], problem.xu[dst + 2])

    @staticmethod
    def _set_active(row, i, turn_on, problem, rs):
        if turn_on:
            row[i] = 1.0
            row[i + 1] = rs.uniform(problem.xl[i + 1], problem.xu[i + 1])
            row[i + 2] = rs.uniform(problem.xl[i + 2], problem.xu[i + 2])
        else:
            row[i] = 0.0


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
            if random_state.random() < 0.5:
                c1, c2 = self._time_slice_swap(a, b, n_steps, random_state)
            else:
                c1, c2 = self._spatial_area_swap(a, b, n_steps, n_slots, problem)
            Q[0, m], Q[1, m] = c1, c2

        return Q

    def _time_slice_swap(self, a, b, n_steps, rs):
        """Cut both parents at the same timestep; swap what comes after.
        A 1-timestep protocol has no valid cut point (nothing to swap), so
        it falls back to returning the parents unchanged rather than crashing."""
        if n_steps < 2:
            return a.copy(), b.copy()

        cut = genome.electrode_start(int(rs.integers(1, n_steps)), 0)
        c1 = np.concatenate([a[:cut], b[cut:]])
        c2 = np.concatenate([b[:cut], a[cut:]])
        return c1, c2

    def _spatial_area_swap(self, a, b, n_steps, n_slots, problem):
        """Split the grid down the middle column; wherever parent A's own slot lands on the far side, swap that slot's genes between children."""
        mid = config.GRID_SIZE / 2.0
        c1, c2 = a.copy(), b.copy()

        for t, (_, col_d) in enumerate(problem.protocol):
            for s in range(n_slots):
                i = genome.electrode_start(t, s)
                if col_d + a[i + 2] >= mid:
                    c1[i:i + 3], c2[i:i + 3] = b[i:i + 3], a[i:i + 3]

        return c1, c2
