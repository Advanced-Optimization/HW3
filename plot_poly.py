import matplotlib.pyplot as plt
import numpy as np


def plot_coordinate_system():
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 4)
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position("zero")
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")
    ax.yaxis.set_ticks_position("left")
    ax.plot(
        (1),
        (0),
        ls="",
        marker=">",
        ms=5,
        color="k",
        transform=ax.get_yaxis_transform(),
        clip_on=False,
    )
    ax.plot(
        (0),
        (1),
        ls="",
        marker="^",
        ms=5,
        color="k",
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )
    ax.set_xlabel(r"$\theta$", loc="right")
    ax.set_ylabel(r"$f(\theta)$", loc="top", rotation=0)
    # ax.set_yticks([])
    return fig, ax


def plot_cost(ts, y, label=None):
    fig, ax = plot_coordinate_system()
    ax.plot(ts, y, label=label)
    ymin = min(-max(y) / 3, min(y))
    ax.set_ylim(ymin, max(y))
    return fig, ax


def plot_global(lifter, ax, sol_dict):
    import itertools

    markers = itertools.cycle(["x", "+", "o", "*"])
    ylim = ax.get_ylim()
    for label, t in sol_dict.items():
        cost = lifter.get_cost(t)
        if ylim[0] < cost < ylim[1]:
            ax.scatter([t], [cost], color="C2", marker=next(markers), label=label)
        elif cost > ylim[1]:
            ax.quiver([t], [ylim[1] - 0.2], [0], [1.0], color="C2", label=None)
            ax.scatter(
                [t], [ylim[1] - 0.2], color="C2", marker=next(markers), label=label
            )
    ax.legend(loc="upper center")


def polish_cost_figure(fig, ax, lifter, title):
    ax.set_title(title, pad=14)
    x0, x1 = lifter.xlims
    pad = 0.1 * (x1 - x0)
    ax.set_xlim(x0 - pad, x1 + pad)
    fig.subplots_adjust(top=0.78)


def plot_calibration_setup(
    rotations,
    landmarks_world,
    landmarks_measured,
    d=3,
    title="Calibration setup in world frame",
    frame_scale=1.5,
):
    import matplotlib.pyplot as plt

    from popcor.popcor.utils.plotting_tools import plot_frame

    # Backward compatibility: allow legacy call with a single rotation matrix.
    if isinstance(rotations, dict):
        rotations_dict = {k: np.asarray(v) for k, v in rotations.items()}
    else:
        rotations_dict = {"estimate": np.asarray(rotations)}

    landmarks_world = np.asarray(landmarks_world)
    landmarks_measured = np.asarray(landmarks_measured)

    if d == 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        frame_points = [np.zeros((2, 1)), frame_scale * np.eye(2)]
        plot_frame(
            ax=ax,
            theta=np.eye(2),
            label="world frame",
            color="k",
            marker="o",
            ls="-",
            alpha=0.9,
            d=2,
            scale=frame_scale,
            r_wc_w=np.zeros(2),
        )

        ax.scatter(
            landmarks_world[0],
            landmarks_world[1],
            color="C0",
            s=70,
            marker="o",
            label="World landmarks",
        )

        for idx, (name, R_cw) in enumerate(rotations_dict.items()):
            color = f"C{idx + 1}"
            ls = "-." if "estimate" in name.lower() else "--"
            marker = "x"

            # Include camera-frame axis endpoints in world coordinates for robust limits.
            frame_points.append(frame_scale * np.asarray(R_cw, dtype=float))

            plot_frame(
                ax=ax,
                theta=R_cw,
                label=f"camera frame ({name})",
                color=color,
                marker=marker,
                ls=ls,
                alpha=0.9,
                d=2,
                scale=frame_scale,
                r_wc_w=np.zeros(2),
            )

            landmarks_reconstructed = R_cw @ landmarks_measured
            ax.scatter(
                landmarks_reconstructed[0],
                landmarks_reconstructed[1],
                color=color,
                s=55,
                marker="x",
                label=f"Reconstructed landmarks ({name})",
            )

            for i in range(landmarks_world.shape[1]):
                ax.plot(
                    [landmarks_world[0, i], landmarks_reconstructed[0, i]],
                    [landmarks_world[1, i], landmarks_reconstructed[1, i]],
                    color=color,
                    linewidth=0.8,
                    alpha=0.35,
                )

        all_point_sets = [landmarks_world]
        all_point_sets.extend([R @ landmarks_measured for R in rotations_dict.values()])
        all_point_sets.extend(frame_points)
        points = np.hstack(all_point_sets)
        mins = points.min(axis=1)
        maxs = points.max(axis=1)
        center = 0.5 * (mins + maxs)
        radius = 0.5 * np.max(maxs - mins) + 0.25
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    elif d == 3:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

        origin = np.zeros(3)
        axes_dirs = np.eye(3) * frame_scale
        axis_colors = ["C0", "C1", "C2"]
        axis_labels = ["x_c", "y_c", "z_c"]
        for direction, color, label in zip(axes_dirs, axis_colors, axis_labels):
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                direction[0],
                direction[1],
                direction[2],
                color=color,
                linewidth=2,
                arrow_length_ratio=0.1,
                label=label,
            )

        ax.scatter(
            landmarks_world[0],
            landmarks_world[1],
            landmarks_world[2],
            color="C0",
            s=70,
            marker="o",
            label="World landmarks",
        )

        for idx, (name, R_cw) in enumerate(rotations_dict.items()):
            color = f"C{idx + 1}"
            landmarks_reconstructed = R_cw @ landmarks_measured

            ax.scatter(
                landmarks_reconstructed[0],
                landmarks_reconstructed[1],
                landmarks_reconstructed[2],
                color=color,
                s=55,
                marker="x",
                label=f"Reconstructed landmarks ({name})",
            )

            for i in range(landmarks_world.shape[1]):
                ax.plot(
                    [landmarks_world[0, i], landmarks_reconstructed[0, i]],
                    [landmarks_world[1, i], landmarks_reconstructed[1, i]],
                    [landmarks_world[2, i], landmarks_reconstructed[2, i]],
                    color=color,
                    linewidth=0.8,
                    alpha=0.35,
                )

        points = np.hstack([landmarks_world] + [R @ landmarks_measured for R in rotations_dict.values()])
        mins = points.min(axis=1)
        maxs = points.max(axis=1)
        center = 0.5 * (mins + maxs)
        radius = 0.5 * np.max(maxs - mins) + 0.5
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=22, azim=35)
    else:
        raise ValueError(f"Unsupported calibration dimension d={d}. Expected 2 or 3.")

    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig, ax


def _as_numpy(matrix):
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    arr = np.asarray(matrix)
    if arr.dtype == object:
        arr = np.asarray(arr, dtype=float)
    return arr


def plot_problem_matrices(Q, A_0, A_known, title=None):
    Q_matrix = _as_numpy(Q)

    A0_raw = A_0[0] if isinstance(A_0, tuple) else A_0
    A_known_raw = A_known[0] if isinstance(A_known, tuple) else A_known

    A_mats = [_as_numpy(A_i) for A_i in list(A0_raw) + list(A_known_raw)]
    mats = [("$Q$", Q_matrix)] + [(f"$A_{i}$", A_i) for i, A_i in enumerate(A_mats)]

    cols = len(mats)
    fig, axs = plt.subplots(1, cols, figsize=(4 * cols, 3.5))
    if cols == 1:
        axs = [axs]

    for ax, (name, matrix) in zip(axs, mats):
        im = ax.imshow(matrix, cmap="coolwarm")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    for name, matrix in mats:
        print(f"{name} =\n{np.array2string(matrix, precision=4)}\n")
