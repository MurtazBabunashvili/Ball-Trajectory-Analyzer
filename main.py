import os
import sys
import argparse
import numpy as np

from data.data_loader import load_dataset, load_game, load_clip, clip_summary
from numerical.interpolation import interpolate_trajectory, cubic_spline, cubic_spline_evaluate
from numerical.finite_differences import compute_velocity, compute_acceleration
from numerical.rootfinding import find_apex, find_landing
from physics.model_fitting import fit_trajectory, compare_models
from physics.vacuum_model import VacuumTrajectory
from physics.drag_model import DragTrajectory
from visualization.plots import dashboard, plot_model_comparison
from visualization.annotate_video import annotate_clip_from_loader

DATASET_ROOT = "Dataset"
OUT_DIR = os.path.join("data", "outputs")
MIN_VISIBLE = 6


def parse_args():
    p = argparse.ArgumentParser(description="Ball Trajectory Analyzer")

    p.add_argument("--dataset", default=DATASET_ROOT, help="Path to Dataset/")
    p.add_argument("--out", default=OUT_DIR, help="Output directory")
    p.add_argument("--game", default=None, help="Process one game (e.g. game1)")
    p.add_argument("--clip", default=None, help="Process one clip (e.g. Clip1); requires --game")

    p.add_argument(
        "--model",
        default="vacuum",
        choices=["vacuum", "drag", "both"],
        help="Physics model to fit"
    )

    p.add_argument(
        "--method",
        default="lm",
        choices=["lm", "gd"],
        help="Optimisation method"
    )

    p.add_argument(
        "--interp",
        default="spline",
        choices=["spline", "hermite", "newton", "lagrange"],
        help="Interpolation method"
    )

    p.add_argument("--no-video", action="store_true", help="Skip annotated video output")
    p.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots")

    return p.parse_args()


def process_clip(clip: dict, args) -> dict | None:
    tag = f"{clip['game']}/{clip['clip']}"

    if clip["n_visible"] < MIN_VISIBLE:
        print(f"  [skip] {tag} — only {clip['n_visible']} visible frames")
        return None

    t, x, y = clip["t"], clip["x"], clip["y"]

    print(f"\n{'─' * 60}")
    print(f"  {clip_summary(clip)}")

    t_full = clip["t_full"]
    t_query = t_full[np.isnan(clip["x_full"])]

    interp_x_fn = interp_y_fn = None

    if len(t_query) > 0 and len(t) >= 2:
        try:
            if args.interp == "hermite":
                dt = np.mean(np.diff(t))

                vx_, vy_ = compute_velocity(x, y, dt, method="richardson")

                xi, yi = interpolate_trajectory(
                    t,
                    x,
                    y,
                    t_query,
                    method="hermite",
                    derivatives={"vx": vx_, "vy": vy_}
                )

            else:
                xi, yi = interpolate_trajectory(t, x, y, t_query, method=args.interp)

            sx = cubic_spline(t, x, mode="natural")
            sy = cubic_spline(t, y, mode="natural")

            interp_x_fn = lambda tt: cubic_spline_evaluate(sx, np.atleast_1d(tt))
            interp_y_fn = lambda tt: cubic_spline_evaluate(sy, np.atleast_1d(tt))

            n_rec = int(np.sum(~np.isnan(xi)))

            print(f"  [interp] {n_rec}/{len(t_query)} missing frames recovered via {args.interp}")

        except Exception as e:
            print(f"  [interp] Warning — interpolation failed: {e}")

    dt = np.mean(np.diff(t)) if len(t) > 1 else 1 / 30

    vx, vy = compute_velocity(x, y, dt, method="richardson")
    ax, ay = compute_acceleration(vx, vy, dt, method="richardson")

    speed = np.sqrt(vx ** 2 + vy ** 2)

    print(f"  [kinematics] peak speed = {speed.max():.1f} px/s  |  mean |ay| = {np.abs(ay).mean():.2f} px/s²")

    apex_t = apex_y = land_t = None

    if len(t) >= 4:
        try:
            sy_spline = cubic_spline(t, vy, mode="natural")

            vy_func = lambda tt: float(cubic_spline_evaluate(sy_spline, [tt]))

            if vy[0] * vy[-1] < 0:
                apex_res = find_apex(vy_func, t[0], t[-1])

                if apex_res.converged:
                    apex_t = apex_res.root

                    sy2 = cubic_spline(t, y, mode="natural")

                    apex_y = float(cubic_spline_evaluate(sy2, [apex_t]))

                    print(f"  [rootfind] apex  t={apex_t:.4f}s  y={apex_y:.1f}px")

            sy2 = cubic_spline(t, y, mode="natural")

            y_func = lambda tt: float(cubic_spline_evaluate(sy2, [tt])) - y[0]

            if y_func(t[0]) * y_func(t[-1]) < 0:
                land_res = find_landing(y_func, t[0], t[-1])

                if land_res.converged:
                    land_t = land_res.root

                    print(f"  [rootfind] land  t={land_t:.4f}s")

        except Exception as e:
            print(f"  [rootfind] Warning — {e}")

    model = None
    fit_vacuum = None
    fit_drag = None

    try:
        if args.model == "both":
            fit_vacuum, fit_drag = compare_models(t, x, y, method=args.method)

            print(
                f"  [fit] vacuum RMSE={fit_vacuum.rmse:.2f}px  "
                f"({'converged' if fit_vacuum.converged else 'NO CONV'})"
            )

            print(
                f"  [fit] drag   RMSE={fit_drag.rmse:.2f}px  "
                f"({'converged' if fit_drag.converged else 'NO CONV'})"
            )

            if fit_drag.rmse < fit_vacuum.rmse:
                model = DragTrajectory(**fit_drag.params)

            else:
                model = VacuumTrajectory(
                    **{
                        k: v
                        for k, v in fit_vacuum.params.items()
                        if k in ("x0", "y0", "vx0", "vy0", "g")
                    }
                )

        else:
            result = fit_trajectory(t, x, y, model=args.model, method=args.method)

            print(
                f"  [fit] {args.model} RMSE={result.rmse:.2f}px  "
                f"({'converged' if result.converged else 'NO CONV'})  "
                f"iters={result.iterations}"
            )

            if args.model == "vacuum":
                model = VacuumTrajectory(
                    **{
                        k: v
                        for k, v in result.params.items()
                        if k in ("x0", "y0", "vx0", "vy0", "g")
                    }
                )

            else:
                model = DragTrajectory(**result.params)

            fit_vacuum = result

    except Exception as e:
        print(f"  [fit] Warning — model fitting failed: {e}")

    if not args.no_plots:
        try:
            history = fit_vacuum.history if fit_vacuum else None

            method_name = f"{args.model}-{args.method}"

            if args.model == "both" and fit_vacuum and fit_drag:
                plot_model_comparison(t, x, y, fit_vacuum, fit_drag)

            else:
                dashboard(
                    t,
                    x,
                    y,
                    model=model,
                    fit_history=history,
                    method_name=method_name
                )

        except Exception as e:
            print(f"  [plots] Warning — {e}")

    if not args.no_video:
        try:
            annotate_clip_from_loader(
                clip,
                dataset_root=args.dataset,
                out_dir=os.path.join(args.out, "videos"),
                model=model,
                interp_x=interp_x_fn,
                interp_y=interp_y_fn,
                velocities={"vx": vx, "vy": vy},
            )

        except Exception as e:
            print(f"  [video] Warning — {e}")

    return {
        "game": clip["game"],
        "clip": clip["clip"],
        "n_visible": clip["n_visible"],
        "n_missing": int(np.sum(np.isnan(clip["x_full"]))),
        "peak_speed": float(speed.max()),
        "apex_t": apex_t,
        "apex_y": apex_y,
        "land_t": land_t,
        "rmse_vacuum": fit_vacuum.rmse if fit_vacuum else None,
        "rmse_drag": fit_drag.rmse if fit_drag else None,
    }


def load_clips(args) -> list[dict]:
    if args.clip:
        if not args.game:
            sys.exit("--clip requires --game")

        clip_path = os.path.join(args.dataset, args.game, args.clip)

        return [load_clip(clip_path)]

    if args.game:
        game_path = os.path.join(args.dataset, args.game)

        return load_game(game_path)

    return load_dataset(args.dataset)


def main():
    args = parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'═' * 60}")
    print(f"  Ball Trajectory Analyzer")
    print(f"  dataset : {args.dataset}")
    print(f"  model   : {args.model}   method : {args.method}")
    print(f"  interp  : {args.interp}")
    print(f"{'═' * 60}")

    clips = load_clips(args)

    if not clips:
        sys.exit(f"No clips found in {args.dataset}")

    print(f"\n  Loaded {len(clips)} clip(s)\n")

    results = []

    for clip in clips:
        rec = process_clip(clip, args)

        if rec:
            results.append(rec)

    print(f"\n{'═' * 60}")
    print(f"  Summary  ({len(results)} clips processed)")
    print(f"{'─' * 60}")
    print(f"  {'Clip':<20} {'Vis':>5} {'Speed':>9} {'RMSE_vac':>10} {'RMSE_drg':>10}")
    print(f"{'─' * 60}")

    for r in results:
        tag = f"{r['game']}/{r['clip']}"

        rmse_v = f"{r['rmse_vacuum']:.2f}" if r['rmse_vacuum'] is not None else "  —  "
        rmse_d = f"{r['rmse_drag']:.2f}" if r['rmse_drag'] is not None else "  —  "

        print(
            f"  {tag:<20} {r['n_visible']:>5} {r['peak_speed']:>8.1f}  "
            f"{rmse_v:>10} {rmse_d:>10}"
        )

    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()