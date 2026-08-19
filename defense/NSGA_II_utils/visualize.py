import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
import defense.NSGA_II_utils.config as config
import defense.NSGA_II_utils.field as field


"""
visualize.py just plots everything
"""


def plot_pareto_front(F, path=None, interactive=False):
    """3D scatter of the front: count, spread, attack_accuracy. interactive=True opens a live, mouse-rotatable window
    instead of saving a file."""
    count, spread, attack_accuracy = -F[:, 0], F[:, 1], F[:, 2]

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


def plot_timesteps(protocol, schedule, path=None, ncols=8, interactive=False):
    """One panel per timestep: droplet in blue, dummy actuations in green."""
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
