import numpy as np
from numerical.edge_detection import canny, gradient, gaussian_kernel, convolve2d


def _candidate_regions(edge_map, min_size=3, max_size=25):
    visited = np.zeros_like(edge_map, dtype=bool)
    H, W    = edge_map.shape
    regions = []

    for i in range(H):
        for j in range(W):
            if edge_map[i, j] == 0 or visited[i, j]:
                continue

            stack  = [(i, j)]
            pixels = []

            while stack:
                r, c = stack.pop()
                if r < 0 or r >= H or c < 0 or c >= W:
                    continue
                if visited[r, c] or edge_map[r, c] == 0:
                    continue
                visited[r, c] = True
                pixels.append((r, c))
                stack.extend([(r+1,c),(r-1,c),(r,c+1),(r,c-1)])

            size = len(pixels)
            if min_size <= size <= max_size:
                rows = [p[0] for p in pixels]
                cols = [p[1] for p in pixels]
                cy   = sum(rows) / size
                cx   = sum(cols) / size
                regions.append((cy, cx, size))

    return regions


def _circularity(edge_map, cy, cx, radius):
    n_samples = 36
    angles    = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    H, W      = edge_map.shape
    hits      = 0

    for a in angles:
        r = int(round(cy + radius * np.sin(a)))
        c = int(round(cx + radius * np.cos(a)))
        if 0 <= r < H and 0 <= c < W and edge_map[r, c] > 0:
            hits += 1

    return hits / n_samples


def locate_ball(gray_frame, sigma = 1.2, t_low = 0.05, t_high = 0.18, min_size = 4, max_size = 30, circularity_thresh = 0.30,):
    edges, magnitude = canny(gray_frame, sigma=sigma, t_low=t_low, t_high=t_high)

    regions = _candidate_regions(edges, min_size=min_size, max_size=max_size)
    if not regions:
        return None

    best_score = -1.0
    best_cx    = None
    best_cy    = None

    for cy, cx, size in regions:
        radius = np.sqrt(size / np.pi)
        score  = _circularity(edges, cy, cx, radius)
        if score > best_score:
            best_score = score
            best_cx    = cx
            best_cy    = cy

    if best_score < circularity_thresh:
        return None

    return best_cx, best_cy


def locate_ball_sequence(frames,fps= 30.0,physical_flip = True,image_height  = 720,**locator_kwargs,):
    t_list = []
    x_list = []
    y_list = []

    for idx, frame in frames:
        if frame.ndim == 3:
            gray = frame.mean(axis=2).astype(np.float32) / 255.0
        else:
            gray = frame.astype(np.float32) / 255.0

        t_list.append(idx / fps)

        result = locate_ball(gray, **locator_kwargs)
        if result is not None:
            cx, cy = result
            if physical_flip:
                cy = image_height - cy
            x_list.append(cx)
            y_list.append(cy)
        else:
            x_list.append(np.nan)
            y_list.append(np.nan)

    t_full = np.array(t_list)
    x_full = np.array(x_list)
    y_full = np.array(y_list)

    visible = ~np.isnan(x_full)
    return t_full[visible], x_full[visible], y_full[visible], t_full, x_full, y_full