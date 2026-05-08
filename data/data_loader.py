import os
import numpy as np
import pandas as pd

FPS = 30.0
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

def load_clip(clip_path: str) -> dict:
    csv_path = os.path.join(clip_path, "Label.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No Label.csv found in {clip_path}")

    df = pd.read_csv(csv_path, sep=None, engine="python")

    df.columns = [c.strip().lower() for c in df.columns]

    col_map = {
        "file name": "filename",
        "visibility": "visibility",
        "x-coordinate": "x_raw",
        "y-coordinate": "y_raw",
        "status": "status"
    }

    df = df.rename(columns=col_map)

    df["frame"] = (df["filename"].str.replace(".jpg", "", case=False).str.strip().astype(int))

    df["t"] = df["frame"] / FPS

    df["x_raw"] = pd.to_numeric(df["x_raw"], errors="coerce")
    df["y_raw"] = pd.to_numeric(df["y_raw"], errors="coerce")

    df["x_phys"] = df["x_raw"]
    df["y_phys"] = IMAGE_HEIGHT - df["y_raw"]

    invisible = df["visibility"] == 0
    x_full = df["x_phys"].copy()
    y_full = df["y_phys"].copy()
    x_full[invisible] = np.nan
    y_full[invisible] = np.nan

    t_full = df["t"].values
    x_full = x_full.values
    y_full = y_full.values

    visible_mask = df["visibility"] > 0
    t_clean = df.loc[visible_mask, "t"].values
    x_clean = df.loc[visible_mask, "x_phys"].values
    y_clean = df.loc[visible_mask, "y_phys"].values

    hits    = df.loc[df["status"] == 1, "t"].values   # racket contact
    bounces = df.loc[df["status"] == 2, "t"].values   # court bounce

    parts = os.path.normpath(clip_path).split(os.sep)
    clip_name = parts[-1]
    game_name = parts[-2]

    return {
        "t": t_clean,
        "x": x_clean,
        "y": y_clean,
        "t_full": t_full,
        "x_full": x_full,
        "y_full": y_full,
        "hits": hits,
        "bounces": bounces,
        "game": game_name,
        "clip": clip_name,
        "n_frames": len(df),
        "n_visible": int(visible_mask.sum()),
    }

def load_game(game_path: str) -> list[dict]:
    clips = []
    entries = sorted(
        os.listdir(game_path),
        key=lambda name: int("".join(filter(str.isdigit, name)) or 0)
    )
    for entry in entries:
        clip_path = os.path.join(game_path, entry)
        if os.path.isdir(clip_path) and entry.lower().startswith("clip"):
            try:
                clips.append(load_clip(clip_path))
            except Exception as e:
                print(f"[data_loader] Warning — skipped {clip_path}: {e}")
    return clips

def load_dataset(dataset_path: str) -> list[dict]:
    all_clips = []
    entries = sorted(
        os.listdir(dataset_path),
        key=lambda name: int("".join(filter(str.isdigit, name)) or 0)
    )
    for entry in entries:
        game_path = os.path.join(dataset_path, entry)
        if os.path.isdir(game_path) and entry.lower().startswith("game"):
            all_clips.extend(load_game(game_path))
    return all_clips

def clip_summary(clip: dict) -> str:
    missing = np.sum(np.isnan(clip["x_full"]))
    return (
        f"{clip['game']}/{clip['clip']} — "
        f"{clip['n_frames']} frames | "
        f"{clip['n_visible']} visible | "
        f"{missing} missing | "
        f"{len(clip['hits'])} hits | "
        f"{len(clip['bounces'])} bounces"
    )
