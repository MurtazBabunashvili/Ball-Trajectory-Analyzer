import os
import cv2
import numpy as np


class FrameExtractor:
    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.video_path = video_path
        self._cap       = cv2.VideoCapture(video_path)

        if not self._cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        self.fps      = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width    = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield idx, idx / self.fps, frame
            idx += 1

    def __len__(self):
        return self.n_frames

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def release(self):
        self._cap.release()

    def read_frame(self, index: int) -> np.ndarray:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self._cap.read()
        if not ret:
            raise IndexError(f"Could not read frame {index}")
        return frame

    def to_gray(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def to_float(self, frame: np.ndarray) -> np.ndarray:
        return frame.astype(np.float32) / 255.0


def save_frames(video_path: str, out_dir: str, prefix: str = "") -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    with FrameExtractor(video_path) as ex:
        for idx, _, frame in ex:
            filename = f"{prefix}{idx:04d}.jpg"
            fpath    = os.path.join(out_dir, filename)
            cv2.imwrite(fpath, frame)
            paths.append(fpath)

    return paths


def load_frames_from_dir(frame_dir: str) -> list[tuple[int, np.ndarray]]:
    files = sorted(
        [f for f in os.listdir(frame_dir) if f.lower().endswith(".jpg")],
        key=lambda f: int(os.path.splitext(f)[0])
    )
    result = []
    for f in files:
        idx   = int(os.path.splitext(f)[0])
        frame = cv2.imread(os.path.join(frame_dir, f))
        if frame is not None:
            result.append((idx, frame))
    return result