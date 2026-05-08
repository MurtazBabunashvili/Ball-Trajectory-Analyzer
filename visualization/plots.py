import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numerical.finite_differences import compute_velocity, compute_acceleration


def plot_trajectory(t, x, y, model=None, title="Trajectory", ax=None):
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(x, y, s=10, c=t, cmap="plasma", label="Detected", zorder=3)

    if model is not None:
        t_dense = np.linspace(t[0], t[-1], 300)
        xm, ym  = model.evaluate(t_dense)
        ax.plot(xm, ym, "w--", linewidth=1.5, label="Model fit")

    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.legend()

    if show:
        plt.tight_layout()
        plt.show()


def plot_velocity(t, x, y, method="central", ax=None):
    show = ax is None
    dt   = np.mean(np.diff(t))
    vx, vy = compute_velocity(x, y, dt, method=method)
    speed  = np.sqrt(vx**2 + vy**2)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(t, vx,    label="vx")
    ax.plot(t, vy,    label="vy")
    ax.plot(t, speed, label="speed", linewidth=2)
    ax.set_title("Velocity")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("px / s")
    ax.legend()

    if show:
        plt.tight_layout()
        plt.show()


def plot_acceleration(t, x, y, method="central", ax=None):
    show = ax is None
    dt   = np.mean(np.diff(t))
    vx, vy = compute_velocity(x, y, dt, method=method)
    ax_, ay = compute_acceleration(vx, vy, dt, method=method)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(t, ax_, label="ax")
    ax.plot(t, ay,  label="ay")
    ax.axhline(0, color="white", linewidth=0.5, linestyle="--")
    ax.set_title("Acceleration")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("px / s²")
    ax.legend()

    if show:
        plt.tight_layout()
        plt.show()


def plot_curvature(t, x, y, method="central", ax=None):
    show = ax is None
    dt   = np.mean(np.diff(t))
    vx, vy = compute_velocity(x, y, dt, method=method)
    ax_, ay = compute_acceleration(vx, vy, dt, method=method)

    denom = (vx**2 + vy**2) ** 1.5
    safe  = denom > 1e-6
    kappa = np.where(safe, np.abs(vx * ay - vy * ax_) / np.where(safe, denom, 1.0), 0.0)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(t, kappa, color="cyan")
    ax.set_title("Curvature κ(t)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("κ (1/px)")

    if show:
        plt.tight_layout()
        plt.show()


def plot_convergence(history, method_name="", ax=None):
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.semilogy(history, linewidth=1.5)
    ax.set_title(f"Convergence — {method_name}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost F(θ)")

    if show:
        plt.tight_layout()
        plt.show()


def plot_residuals(t, x, y, model, ax=None):
    show = ax is None
    xp, yp = model.evaluate(t)
    rx = x - xp
    ry = y - yp

    if ax is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4))
    else:
        axes = ax

    axes[0].stem(t, rx)
    axes[0].set_title("Residuals x")
    axes[0].set_xlabel("t (s)")
    axes[0].set_ylabel("x_obs − x_pred (px)")

    axes[1].stem(t, ry)
    axes[1].set_title("Residuals y")
    axes[1].set_xlabel("t (s)")
    axes[1].set_ylabel("y_obs − y_pred (px)")

    if show:
        plt.tight_layout()
        plt.show()


def plot_model_comparison(t, x, y, res_vacuum, res_drag):
    from physics.vacuum_model import VacuumTrajectory
    from physics.drag_model   import DragTrajectory

    vac_model  = VacuumTrajectory(**{k: v for k, v in res_vacuum.params.items()})
    drag_model = DragTrajectory(**{k: v for k, v in res_drag.params.items()})

    t_dense    = np.linspace(t[0], t[-1], 300)
    xv, yv     = vac_model.evaluate(t_dense)
    xd, yd     = drag_model.evaluate(t_dense)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(x, y, s=12, color="white", zorder=4, label="Detected")
    ax.plot(xv, yv, "--", color="orange", label=f"Vacuum  RMSE={res_vacuum.rmse:.1f}px")
    ax.plot(xd, yd, "-",  color="cyan",   label=f"Drag    RMSE={res_drag.rmse:.1f}px")
    ax.set_title("Vacuum vs Drag Model")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def dashboard(t, x, y, model, fit_history=None, method_name=""):
    has_history = fit_history is not None
    n_cols = 2
    n_rows = 3 if has_history else 2

    fig = plt.figure(figsize=(14, 5 * n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig)

    ax_traj = fig.add_subplot(gs[0, :])
    plot_trajectory(t, x, y, model=model, ax=ax_traj)

    ax_vel = fig.add_subplot(gs[1, 0])
    plot_velocity(t, x, y, ax=ax_vel)

    ax_acc = fig.add_subplot(gs[1, 1])
    plot_acceleration(t, x, y, ax=ax_acc)

    if has_history:
        ax_conv = fig.add_subplot(gs[2, 0])
        plot_convergence(fit_history, method_name=method_name, ax=ax_conv)

        ax_curv = fig.add_subplot(gs[2, 1])
        plot_curvature(t, x, y, ax=ax_curv)

    plt.tight_layout()
    plt.show()