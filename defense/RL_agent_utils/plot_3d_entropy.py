import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_3d_entropy_path(true_path, grid_size, obstacles, entropy_values):
    """
    3D line plot of the true droplet path over time, with red scatter at each timestep spread out by that timestep's own entropy value (high entropy -> wide spread).
    """

    x = [pos[0] for pos in true_path]
    y = [pos[1] for pos in true_path]

    fig = plt.figure(figsize=(10, 6), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xticks(np.arange(0, grid_size + 1, 1))
    ax.set_xticklabels([str(i) if i % 2 == 0 else "" for i in range(grid_size + 1)])
    ax.set_yticks(np.arange(0, grid_size + 1, 1))
    ax.set_yticklabels([str(i) if i % 2 == 0 else "" for i in range(grid_size + 1)])
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_zlim(0, 1)

    ax.set_box_aspect((1, 1, 0.75))
    ax.view_init(elev=20, azim=-135)

    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["linestyle"] = ":"
        axis._axinfo["grid"]["linewidth"] = 0.8

    ax.computed_zorder = False
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    for r, c in obstacles:
        verts = [[(r, c, 0), (r+1, c, 0), (r+1, c+1, 0), (r, c+1, 0)]]
        ax.add_collection3d(Poly3DCollection(verts, facecolors='black', alpha=1.0))

    t = np.linspace(0, 1, len(x))

    x_vis = np.array(x) + 0.5
    y_vis = np.array(y) + 0.5

    ax.plot(x_vis, y_vis, 0, color='gray', alpha=0.75, lw=2)

    entropy_values = np.abs(np.array(entropy_values))

    for i in range(len(x)-1):
        ax.plot(x_vis[i:i+2], y_vis[i:i+2], t[i:i+2], color=plt.cm.viridis(t[i]), lw=3)

    for i in range(len(entropy_values)):
        sigma = entropy_values[i] * 2.0
        effective_sigma = max(sigma, 0.02)
        noise_x, noise_y = np.random.normal(0, effective_sigma, 15), np.random.normal(0, effective_sigma, 15)

        ax.scatter(x_vis[i] + noise_x, y_vis[i] + noise_y, t[i],
                   color='red', s=40, alpha=0.5, marker='o', edgecolors='none')

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    plt.show()
