import os
import random
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
os.chdir(os.path.join(_REPO, "defense"))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "environment_utils"))
sys.path.insert(0, os.path.join(_REPO, "defense", "RL_agent_utils"))

from attack_agent import DRQNagent
from belief_state import BeliefStateMap
from CustomFeatureExtractor import CustomFeatureExtractor
from Env import VoltageDistributionEnv
from multipass_refinement import MultipassEnvWrapper, PassBasedMasking, extract_qvalues

from defense.NSGA_II_utils import config, field, genome

DATA = os.path.join(_HERE, "multi_protocol_defense.npz")
PLOT = os.path.join(_HERE, "multipass_attacker_entropy.png")
MODEL = os.path.join(_REPO, "models", "RecurrentDQN_Finalized_SCIBOT_V1.pkl")
PASSES = 3

"""
Reads multi_protocol_defense.npz. For each protocol it picks the defense with the lowest surrogate attack accuracy, then runs the trained attacker through the 3-pass
refinement `n_runs` times, once with the defense and once without, recording the attacker's belief entropy at every timestep.

The attacker's very first probe of pass 1 is chosen at random (it has no information yet)

  plot()                 # 10 runs per protocol
  plot(n_runs=20)        # more runs
"""


class _Wrapper(MultipassEnvWrapper):
    """Runs the fixed protocol, injects the real+dummy field into what the attacker sees, and tracks a separate 'ground-truth' belief scored only on the true droplet field."""

    def __init__(self, env, pbm, pass_num, schedule, protocol):
        super().__init__(env, pbm, pass_num, stored_protocol=protocol)
        self.schedule = schedule
        self.gt_belief = BeliefStateMap(env.size, env.belief.sigma, env.belief.threshold)
        self.entropy = []

    def _apply_defense(self):
        t = self.env._droplet_path_index
        dummies = self.schedule[t] if t < len(self.schedule) else []
        self.env._voltage_grid = field.combined_field(tuple(self.env._droplet_location), dummies)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.env._droplet_path = self.stored_protocol
        self.env._droplet_path_index = 0
        self.env._droplet_location = self.env._droplet_path[0]
        self.env.end_pos = self.env._droplet_path[-1]
        self.env.length = len(self.env._droplet_path)
        self._apply_defense()
        if self.pass_num > 1:
            info["action_mask"] = self.pass_based_masking.get_refined_mask(0, info["action_mask"])
        self.timestep = 0
        self.pass_based_masking.start_new_episode()
        self.gt_belief.reset()
        self.entropy = []
        return obs, info

    def step(self, action):
        true_cell = tuple(self.env._droplet_location)
        obs, reward, terminated, truncated, info = super().step(action)
        probe = (action // self.grid_size, action % self.grid_size)
        self.gt_belief.prediction()
        self.gt_belief.correction(measurement=field.real_field(true_cell)[probe], probe_position=probe)
        self.entropy.append(self.gt_belief.get_state_entropy())   # already normalized to [0, 1]
        self._apply_defense()
        return obs, reward, terminated, truncated, info


def _one_rollout(agent, protocol, schedule):
    
    """One 3-pass rollout. Returns a list of passes entropy arrays (native length per pass)."""

    pbm = PassBasedMasking(config.GRID_SIZE)
    per_pass = []
    for pass_num in range(1, PASSES + 1):
        env = _Wrapper(VoltageDistributionEnv(), pbm, pass_num, schedule, protocol)
        obs, info = env.reset()
        hidden, done = None, False
        while not done:
            mask = info.get("action_mask", None)
            q, hidden = extract_qvalues(agent, obs, hidden)
            valid = np.where(np.array(mask) == 1)[0] if mask is not None else np.arange(config.GRID_SIZE ** 2)
            if pass_num == 1 and env.timestep == 0 and len(valid):
                action = int(np.random.choice(valid))            # zero-information first probe
            elif len(valid):
                action = int(valid[np.argmax(q[valid])])
            else:
                action = int(np.argmax(q))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        per_pass.append(np.array(env.entropy, float))
    return per_pass


def plot(n_runs=10, points_per_pass=20, seed=0, data=DATA, out=PLOT):
    d = np.load(data, allow_pickle=True)
    protocols, X, F = d["protocols"], d["X"], d["F"]
    n_proto = len(protocols)

    random.seed(seed)
    np.random.seed(seed)
    agent = DRQNagent.load_complete(
        MODEL,
        env_constructor=lambda: VoltageDistributionEnv(),
        feature_extractor_constructor=lambda s: CustomFeatureExtractor(s),
    )

    # common normalized-timestep grid so protocols of different lengths line up
    grid = np.linspace(0, 1, points_per_pass, endpoint=False)
    x_axis = np.concatenate([grid + p for p in range(PASSES)])

    defended, undefended = [], []      # one row per (protocol, run)
    for k in range(n_proto):
        protocol = [tuple(int(v) for v in c) for c in protocols[k]]
        idx = int(np.argmin(F[k][:, 2]))                          # lowest surrogate attack accuracy
        schedule = genome.decode(X[k][idx], protocol)
        empty = [[] for _ in protocol]
        print(f"[{k + 1}/{n_proto}] protocol length {len(protocol)}, "
              f"picked defense idx {idx}, attack_accuracy {F[k][idx, 2]:.3f}")

        for _ in range(n_runs):
            for sched, bucket in ((schedule, defended), (empty, undefended)):
                per_pass = _one_rollout(agent, protocol, sched)
                segments = []
                for p in per_pass:
                    if len(p) > 1:
                        segments.append(np.interp(grid, np.linspace(0, 1, len(p), endpoint=False), p))
                    else:
                        segments.append(np.full(points_per_pass, p[0] if len(p) else 0.0))
                bucket.append(np.concatenate(segments))

    defended = np.array(defended).reshape(n_proto, n_runs, -1).mean(axis=1)      # (n_proto, T)
    undefended = np.array(undefended).reshape(n_proto, n_runs, -1).mean(axis=1)

    c_undef, c_def = "#d1495b", "#2e6f95"
    fig, ax = plt.subplots(figsize=(11, 6))
    for data_arr, color, label in ((undefended, c_undef, "no defense"), (defended, c_def, "defense")):
        for protocol_curve in data_arr:                                         # one faint line per protocol
            ax.plot(x_axis, protocol_curve, color=color, lw=0.8, alpha=0.22, zorder=1)
        mean = data_arr.mean(0)
        lo, hi = np.percentile(data_arr, [25, 75], axis=0)                       # inter-protocol IQR
        ax.fill_between(x_axis, lo, hi, color=color, alpha=0.20, zorder=2)
        ax.plot(x_axis, mean, color=color, lw=2.8, label=label, zorder=4)

    for xb in (1, 2):
        ax.axvline(xb, color="0.35", ls="--", lw=1)
    for i, xc in enumerate((0.5, 1.5, 2.5), 1):
        ax.text(xc, 0.965, f"Pass {i}", ha="center", va="top", fontsize=11, color="0.3")

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Normalized Timesteps", fontsize=13)
    ax.set_ylabel("Entropy", fontsize=13)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=12, loc="upper right", bbox_to_anchor=(0.99, 0.92))
    fig.text(0.5, 1.06, "Multipass Attacker Entropy: Defended vs Undefended",
             ha="center", fontsize=16, fontweight="bold")
    fig.text(0.5, 1.00,
             f"({n_proto} protocols, lowest attack-accuracy defense, {n_runs} runs each; band = inter-protocol IQR)",
             ha="center", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved {out}")
    return out


if __name__ == "__main__":
    plot()
