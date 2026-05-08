import os
import cv2
import numpy as np

IMAGE_HEIGHT = 720
FPS = 30.0

COL_BALL = (0, 255, 255)
COL_INTERP = (0, 165, 255)
COL_MODEL = (255, 255, 0)
COL_HIT = (0, 255, 0)
COL_BOUNCE = (0, 0, 255)
COL_VELOCITY = (255, 0, 255)
COL_TEXT = (255, 255, 255)


def _phys_to_px(x, y, h=IMAGE_HEIGHT):
    return int(round(x)), int(round(h - y))


def _draw_trail(frame, positions_px, max_trail=8):
    n = len(positions_px)

    for i, (px, py) in enumerate(positions_px):
        alpha = (i + 1) / n
        radius = max(2, int(4 * alpha))
        colour = tuple(int(c * alpha) for c in COL_BALL)
        cv2.circle(frame, (px, py), radius, colour, -1, cv2.LINE_AA)


def _draw_velocity_arrow(frame, cx, cy, vx, vy, scale=0.05):
    ex = int(round(cx + vx * scale))
    ey = int(round(cy - vy * scale))

    cv2.arrowedLine(frame, (cx, cy), (ex, ey), COL_VELOCITY, 2, cv2.LINE_AA, tipLength=0.3)


def _draw_model_curve(frame, model, t_start, t_end, n_pts=120, h=IMAGE_HEIGHT):
    ts = np.linspace(t_start, t_end, n_pts)
    xs, ys = model.evaluate(ts)

    pts = np.array([_phys_to_px(x, y, h) for x, y in zip(xs, ys)], dtype=np.int32)

    H, W = frame.shape[:2]

    mask = (
        (pts[:, 0] >= 0) &
        (pts[:, 0] < W) &
        (pts[:, 1] >= 0) &
        (pts[:, 1] < H)
    )

    pts = pts[mask]

    if len(pts) >= 2:
        cv2.polylines(frame, [pts], False, COL_MODEL, 2, cv2.LINE_AA)


def _draw_event_marker(frame, px, py, label, colour, radius=10):
    cv2.circle(frame, (px, py), radius, colour, 2, cv2.LINE_AA)

    cv2.putText(
        frame,
        label,
        (px + radius + 3, py + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        colour,
        1,
        cv2.LINE_AA
    )


def _draw_hud(frame, t, speed_px_s, rmse=None):
    lines = [
        f"t = {t:.3f} s",
        f"speed = {speed_px_s:.1f} px/s",
    ]

    if rmse is not None:
        lines.append(f"model RMSE = {rmse:.2f} px")

    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (12, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_TEXT, 1, cv2.LINE_AA)


def annotate_clip(clip: dict, frame_dir: str, out_path: str, model=None, interp_x=None, interp_y=None, velocities=None, trail_len: int = 8, fps: float = FPS, image_height: int = IMAGE_HEIGHT, event_radius_s: float = 0.10):

    files = sorted(
        [f for f in os.listdir(frame_dir) if f.lower().endswith(".jpg")],
        key=lambda f: int(os.path.splitext(f)[0])
    )

    if not files:
        raise FileNotFoundError(f"No .jpg frames found in {frame_dir}")

    sample = cv2.imread(os.path.join(frame_dir, files[0]))
    H, W = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    visible_lookup = {
        int(round(t * fps)): (x, y)
        for t, x, y in zip(clip["t"], clip["x"], clip["y"])
    }

    t_full = clip["t_full"]
    x_full = clip["x_full"]
    y_full = clip["y_full"]
    n_full = len(t_full)

    trail_px = []

    for file_idx, fname in enumerate(files):
        frame_no = int(os.path.splitext(fname)[0])
        t_now = frame_no / fps

        frame = cv2.imread(os.path.join(frame_dir, fname))

        if frame is None:
            continue

        if model is not None:
            t0 = clip["t_full"][0]
            t1 = clip["t_full"][-1]
            _draw_model_curve(frame, model, t0, t1, h=image_height)

        for t_hit in clip["hits"]:
            if abs(t_now - t_hit) <= event_radius_s:

                if frame_no in visible_lookup:
                    xp, yp = visible_lookup[frame_no]
                    px, py = _phys_to_px(xp, yp, image_height)
                else:
                    px, py = W // 2, H // 2

                _draw_event_marker(frame, px, py, "HIT", COL_HIT)

        for t_bnc in clip["bounces"]:
            if abs(t_now - t_bnc) <= event_radius_s:

                if frame_no in visible_lookup:
                    xp, yp = visible_lookup[frame_no]
                    px, py = _phys_to_px(xp, yp, image_height)
                else:
                    px, py = W // 2, H // 2

                _draw_event_marker(frame, px, py, "BOUNCE", COL_BOUNCE)

        speed_now = 0.0

        if frame_no in visible_lookup:
            xp, yp = visible_lookup[frame_no]
            px, py = _phys_to_px(xp, yp, image_height)

            trail_px.append((px, py))

            if len(trail_px) > trail_len:
                trail_px.pop(0)

            _draw_trail(frame, trail_px)

            cv2.circle(frame, (px, py), 6, COL_BALL, -1, cv2.LINE_AA)

            if velocities is not None:
                idx = int(np.argmin(np.abs(clip["t"] - t_now)))

                vx_now = velocities["vx"][idx]
                vy_now = velocities["vy"][idx]

                speed_now = np.sqrt(vx_now ** 2 + vy_now ** 2)

                _draw_velocity_arrow(frame, px, py, vx_now, vy_now)

        elif interp_x is not None and interp_y is not None:
            try:
                xi = float(interp_x(t_now))
                yi = float(interp_y(t_now))

                px, py = _phys_to_px(xi, yi, image_height)

                if 0 <= px < W and 0 <= py < H:
                    cv2.circle(frame, (px, py), 5, COL_INTERP, -1, cv2.LINE_AA)

                    trail_px.append((px, py))

                    if len(trail_px) > trail_len:
                        trail_px.pop(0)

            except Exception:
                pass

        rmse_val = model.rmse(clip["t"], clip["x"], clip["y"]) if model else None

        _draw_hud(frame, t_now, speed_now, rmse=rmse_val)

        writer.write(frame)

    writer.release()

    print(f"[annotate_video] Saved → {out_path}")


def annotate_clip_from_loader(clip: dict, dataset_root: str, out_dir: str, **kwargs):
    frame_dir = os.path.join(dataset_root, clip["game"], clip["clip"])

    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{clip['game']}_{clip['clip']}.mp4")

    annotate_clip(clip, frame_dir, out_path, **kwargs)