import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
import defense.NSGA_II_utils.config as config
import defense.NSGA_II_utils.field as field


"""
visualize.py just plots everything
"""


def plot_pareto_front(F, path=None, interactive=False):
    """
    3D scatter of the front: count, spread, attack_accuracy. interactive=True opens a live, mouse-rotatable window instead of saving a file.
    """
    count, spread, attack_accuracy = F[:, 0], F[:, 1], F[:, 2]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(count, spread, attack_accuracy, c=attack_accuracy, cmap='viridis_r',
                    s=25, alpha=0.85, depthshade=True, picker=True, pickradius=6)

    ax.set_xlabel('count fitness [Minimization]')
    ax.set_ylabel('spread fitness [Minimization]')
    ax.set_zlabel('attack accuracy [Minimization]')
    ax.set_title(f'Pareto front  --  {len(F)} points  (click a point for its index)')
    ax.view_init(elev=22, azim=-60)
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label='attack accuracy')
    fig.tight_layout()

    if interactive:
        def on_pick(event):
            for idx in event.ind:
                print(f"index {idx}:  count={count[idx]:.3f}  spread={spread[idx]:.3f}  "
                     f"attack_accuracy={attack_accuracy[idx]:.3f}   ->  python3 run_defense.py --pick {idx}")

        fig.canvas.mpl_connect('pick_event', on_pick)
        plt.show()
    elif path:
        fig.savefig(path, dpi=130, bbox_inches='tight')
        plt.close(fig)
    return fig


def _frontier_line(x, y):
    """2D non-dominated staircase of (x,y), both minimized, sorted for a clean line."""
    order = np.argsort(x)
    frontier_x, frontier_y, best_y = [], [], np.inf
    for xi, yi in zip(x[order], y[order]):
        if yi < best_y:
            frontier_x.append(xi)
            frontier_y.append(yi)
            best_y = yi
    return frontier_x, frontier_y


def plot_pareto_slice(F, pop_size, n_gen):
    """
    Single 2D projection of the 3-objective front: actuation cost (x) vs routing-area cost (y),
    each point colored by its surrogate attack accuracy.
    """
    count, spread, attack = F[:, 0], F[:, 1], F[:, 2]

    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(count, spread, c=attack, cmap='viridis_r', s=18, alpha=0.8, label='front points')
    frontier_x, frontier_y = _frontier_line(count, spread)
    ax.plot(frontier_x, frontier_y, color='black', lw=1.5, marker='o', markersize=3, label='2D frontier')

    y_top = spread.max() * 1.05
    ax.fill_between(list(frontier_x) + [count.max()], list(frontier_y) + [frontier_y[-1]],
                    y_top, color='0.5', alpha=0.125, zorder=0)
    ax.set_ylim(spread.min() - 0.002, y_top)

    ax.set_xlabel('Actuation Cost',fontsize=14)
    ax.set_ylabel('Routing-Area Cost',fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Surrogate Attack Accuracy",labelpad=15, fontsize=14)

    fig.text(0.5, 1.08, 'Pareto Front of Genetically Evolved Defense',
              ha='center', fontsize=16, fontweight='bold')
    fig.text(0.5, 1.02, f'(Population Size: {pop_size}, Number of Generations: {n_gen})',
              ha='center', fontsize=14)
    fig.tight_layout()


def plot_protocol_sample(protocol, schedule, n=9, ncols=3, path=None, interactive=False):
    """
    3x3 figure: n timesteps sampled at random (sorted), droplet in blue, dummy actuations in green. 
    """
    rng = np.random.default_rng()
    n = min(n, len(protocol))
    picks = sorted(rng.choice(len(protocol), size=n, replace=False).tolist())

    fields = field.field_sequence(protocol, schedule)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.1 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, t in zip(axes, picks):
        cell, dummies, measured = protocol[t], schedule[t], fields[t]

        ax.imshow(measured, cmap='inferno', vmin=0.0, vmax=1.0, origin='upper')
        for row, col in config.OBSTACLES:
            ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                       facecolor='0.35', edgecolor='none'))

        ax.plot(cell[1], cell[0], 'o', color='deepskyblue', markersize=7,
                markeredgecolor='white', markeredgewidth=0.8)
        for row, col in dummies:
            ax.plot(col, row, 's', color='lime', markersize=6, alpha=0.95,
                    markeredgecolor='white', markeredgewidth=0.6)

        ax.set_title(f't = {t}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n:]:
        ax.axis('off')

    fig.text(0.5, 1.06, 'Protocol and Dummy Actuation Defense',
             ha='center', fontsize=16, fontweight='bold')
    fig.text(0.5, 1.00, f'(droplet = blue, dummy = green; {n} timesteps sampled at random)',
             ha='center', fontsize=12)
    fig.tight_layout()

    if interactive:
        plt.show()
    elif path:
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_timesteps(protocol, schedule, path=None, ncols=8, interactive=False):
    """
    One panel per timestep: droplet in blue, dummy actuations in green.
    """
    fields = field.field_sequence(protocol, schedule)
    nrows = int(np.ceil(len(protocol) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(1.7 * ncols, 1.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for t, (cell, dummies, measured) in enumerate(zip(protocol, schedule, fields)):
        ax = axes[t]

        ax.imshow(measured, cmap='inferno', vmin=0.0, vmax=1.0, origin='upper')
        for row, col in config.OBSTACLES:
            ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                       facecolor='0.35', edgecolor='none'))

        ax.plot(cell[1], cell[0], 'o', color='deepskyblue', markersize=6,
                markeredgecolor='white', markeredgewidth=0.8)
        for row, col in dummies:
            ax.plot(col, row, 's', color='lime', markersize=5, alpha=0.95,
                    markeredgecolor='white', markeredgewidth=0.6)

        ax.set_title(f't={t}  ({len(dummies)})', fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(protocol):]:
        ax.axis('off')

    fig.suptitle('Protocol and dummy actuations per timestep '
                 '(blue = droplet, green = dummy)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if interactive:
        plt.show()
    elif path:
        fig.savefig(path, dpi=130, bbox_inches='tight')
        plt.close(fig)
    return fig
