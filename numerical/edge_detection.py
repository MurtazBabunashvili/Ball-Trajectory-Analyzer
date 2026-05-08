#Implemented methods:
# 1. Differentiation matrix
# 2. 1D / 2D Discrete convolutions
# 3. Separable convolution
# 4. Two-step separable convolution
# 5. edge point
# 6. Gaussian, Sobel, Prewitt, Scharr, Laplacian kernels


import numpy as np

#Differentitation matrix
def differentiation_matrix(n, h=1.0):
    D = np.zeros((n-1, n))
    for i in range(n-1):
        D[i,i] = -1.0
        D[i, i+1] = 1.0
    return D/h


#Convolutions

def convolve2d(image, kernel):
    H, W = image.shape
    m, n = kernel.shape

    pad_r = m//2
    pad_c = n //2

    padded = np.pad(image, ((pad_r, pad_r), (pad_c, pad_c)), mode='constant')
    output = np.zeros_like(image, dtype=float)

    for i in range(H):
        for j in range(W):
            output[i, j] = np.sum(kernel * padded[i:i+m, j:j+n])
    return output

def convolve2d_separable(image, row_kernel, col_kernel):
    temp = convolve_1d(image, col_kernel.ravel(), axis=0)
    output = convolve_1d(temp, row_kernel.ravel(), axis=1)

    return output

def convolve_1d(image, kernel_1d, axis):
    m = len(kernel_1d)
    pad = m//2
    output = np.zeros_like(image, dtype=float)

    if axis == 0:  # vertical
        padded = np.pad(image, ((pad, pad), (0, 0)), mode='constant')
        for i in range(image.shape[0]):
            output[i, :] = np.sum(
                kernel_1d[:, None] * padded[i:i + m, :], axis=0
            )
    else:  # horizontal
        padded = np.pad(image, ((0, 0), (pad, pad)), mode='constant')
        for j in range(image.shape[1]):
            output[:, j] = np.sum(
                kernel_1d[None, :] * padded[:, j:j + m], axis=1
            )
    return output

#Kernels
def gaussian_kernel(size, sigma):
    if size %2 == 0:
        raise ValueError("Kernel size must be odd")

    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy= np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)

    return kernel / kernel.sum() #return normalized answer

def sobel_kernel():
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=float)
    Gy = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=float)
    return Gx, Gy

def prewitt_kernel():
    Gx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=float)
    Gy = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]], dtype=float)
    return Gx, Gy

def scharr_kernel():
    Gx = np.array([[ -3, 0,  3],
                   [-10, 0, 10],
                   [ -3, 0,  3]], dtype=float)
    Gy = np.array([[-3, -10, -3],
                   [ 0,   0,  0],
                   [ 3,  10,  3]], dtype=float)
    return Gx, Gy

def laplacian_kernel():
    return np.array([[0,  1, 0],
                     [1, -4, 1],
                     [0,  1, 0]], dtype=float)

def laplacian_of_gaussian_kernel(size, sigma):
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    ax     = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    r2     = xx**2 + yy**2

    kernel = -(1 / (np.pi * sigma**4)) * (1 - r2 / (2 * sigma**2)) * np.exp(-r2 / (2 * sigma**2))
    kernel -= kernel.mean()

    return kernel

#Gradient and edge magnitude

def gradient(image, kernel="sobel"):
    kernels = {"sobel": sobel_kernel,
               "prewitt": prewitt_kernel,
               "scharr": scharr_kernel}
    if kernel not in kernels:
        raise ValueError(f"Unknown kernel: {kernel}. Choose: sobel | prewitt | scharr")

    Kx, Ky = kernels[kernel]()
    Gx = convolve2d(image, Kx)
    Gy = convolve2d(image, Ky)

    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    direction = np.arctan2(Gy, Gx)

    return magnitude, direction, Gx, Gy

def smooth_then_gradient(image, sigma=1.0, kernel_size=5, deriv_kernel="sobel"):
    G       = gaussian_kernel(kernel_size, sigma)

    g_1d    = G[kernel_size // 2, :]
    smoothed = convolve2d_separable(image, g_1d.reshape(1, -1), g_1d.reshape(-1, 1))

    return gradient(smoothed, kernel=deriv_kernel)

def threshold(magnitude, thresh):
    return (magnitude > thresh).astype(np.uint8)

#Canny edge detector
def non_maximum_suppression(magnitude, direction):
    H, W = magnitude.shape
    output = np.zeros_like(magnitude)
    angle = np.degrees(direction) % 180

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            a = angle[i, j]

            if (0 <= a < 22.5) or (157.5 <= a < 180):
                p, q = magnitude[i, j - 1], magnitude[i, j + 1]
            elif 22.5 <= a < 67.5:
                p, q = magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]
            elif 67.5 <= a < 112.5:
                p, q = magnitude[i - 1, j], magnitude[i + 1, j]
            else:
                p, q = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]

            if magnitude[i, j] >= p and magnitude[i, j] >= q:
                output[i, j] = magnitude[i, j]

    return output

def hysteresis_threshold(nms, t_low, t_high):
    strong = (nms >= t_high)
    weak   = (nms >= t_low) & (nms < t_high)
    output = np.zeros_like(nms, dtype=np.uint8)
    output[strong] = 1

    changed = True
    while changed:
        changed = False
        for i in range(1, nms.shape[0] - 1):
            for j in range(1, nms.shape[1] - 1):
                if weak[i, j] and output[i-1:i+2, j-1:j+2].any():
                    output[i, j] = 1
                    weak[i, j]   = False
                    changed       = True

    return output

def canny(image, sigma=1.0, t_low=0.05, t_high=0.15, kernel_size=5):
    magnitude, direction, _, _ = smooth_then_gradient(
        image, sigma=sigma, kernel_size=kernel_size, deriv_kernel="sobel"
    )

    mag_norm = magnitude / (magnitude.max() + 1e-10)
    nms      = non_maximum_suppression(mag_norm, direction)
    edges    = hysteresis_threshold(nms, t_low, t_high)

    return edges, magnitude

#Laplacian of gaussian
def log_edge_detect(image, sigma=1.0, size=9, threshold=0.01):
    K            = laplacian_of_gaussian_kernel(size, sigma)
    log_response = convolve2d(image, K)

    H, W = log_response.shape
    edges = np.zeros((H, W), dtype=np.uint8)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            patch = log_response[i - 1:i + 2, j - 1:j + 2]
            if patch.min() < 0 < patch.max():
                if np.abs(log_response[i, j]) < threshold:
                    edges[i, j] = 1

    return edges, log_response