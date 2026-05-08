import numpy as np
from physics.vacuum_model import VacuumTrajectory
from physics.drag_model   import DragTrajectory


class FitResult:
    def __init__(self, params, rmse, iterations, converged, history, method):
        self.params     = params      # dict  — final fitted parameters
        self.rmse       = rmse        # float — root mean squared error
        self.iterations = iterations  # int
        self.converged  = converged   # bool
        self.history    = history     # list[float] — cost per iteration
        self.method     = method      # str

    def __repr__(self):
        status = "converged" if self.converged else "did not converge"
        return (f"FitResult(method={self.method}, rmse={self.rmse:.4f}, "
                f"iters={self.iterations}, {status})\n"
                f"  params={self.params}")



def _build_vacuum(theta):
    return VacuumTrajectory(theta[0], theta[1], theta[2], theta[3])

def _build_drag(theta):
    return DragTrajectory(theta[0], theta[1], theta[2], theta[3], km=theta[4])

_BUILDERS = {
    "vacuum": (_build_vacuum, 4),   # (builder_fn, n_params)
    "drag":   (_build_drag,   5),
}

def _residuals(theta, t_obs, x_obs, y_obs, builder):
    try:
        model = builder(theta)
        return model.residuals(t_obs, x_obs, y_obs)
    except Exception:
        return np.full(2 * len(t_obs), 1e6)

def _cost(r):
    return 0.5 * np.dot(r, r)


def _jacobian_fd(theta, t_obs, x_obs, y_obs, builder, h=1e-5):
    n      = len(theta)
    r0     = _residuals(theta, t_obs, x_obs, y_obs, builder)
    m      = len(r0)
    J      = np.zeros((m, n))
    for j in range(n):
        e        = np.zeros(n)
        e[j]     = h
        r_plus   = _residuals(theta + e, t_obs, x_obs, y_obs, builder)
        r_minus  = _residuals(theta - e, t_obs, x_obs, y_obs, builder)
        J[:, j]  = (r_plus - r_minus) / (2 * h)
    return J

def _gauss_newton_step(J, r):
    JtJ = J.T @ J
    Jtr = J.T @ r
    try:
        delta = np.linalg.solve(JtJ, -Jtr)
    except np.linalg.LinAlgError:
        delta = -Jtr
    return delta

def levenberg_marquardt(
    t_obs, x_obs, y_obs,
    theta0,
    builder,
    tol        = 1e-6,
    max_iter   = 200,
    lam0       = 1e-2,
    lam_up     = 10.0,
    lam_dn     = 0.1,
    rho_good   = 0.25,
):
    theta   = np.array(theta0, dtype=float)
    lam     = float(lam0)
    history = []

    for k in range(max_iter):
        r    = _residuals(theta, t_obs, x_obs, y_obs, builder)
        cost = _cost(r)
        history.append(cost)

        J    = _jacobian_fd(theta, t_obs, x_obs, y_obs, builder)
        JtJ  = J.T @ J
        Jtr  = J.T @ r

        grad_norm = np.linalg.norm(Jtr)
        if grad_norm < tol:
            return FitResult(_vec_to_params(theta, builder), _rmse(r, len(t_obs)),
                             k + 1, True, history, "levenberg-marquardt")

        A     = JtJ + lam * np.eye(len(theta))
        try:
            delta = np.linalg.solve(A, -Jtr)
        except np.linalg.LinAlgError:
            lam  *= lam_up
            continue

        theta_new = theta + delta
        r_new     = _residuals(theta_new, t_obs, x_obs, y_obs, builder)
        cost_new  = _cost(r_new)

        predicted = _cost(r + J @ delta)
        actual    = cost - cost_new
        rho       = actual / (cost - predicted + 1e-14)

        if rho > rho_good:          # good step — accept, relax damping
            theta = theta_new
            lam   = max(lam * lam_dn, 1e-12)
        else:                        # bad step — reject, increase damping
            lam  *= lam_up

        if np.linalg.norm(delta) < tol * (np.linalg.norm(theta) + tol):
            r_final = _residuals(theta, t_obs, x_obs, y_obs, builder)
            return FitResult(_vec_to_params(theta, builder), _rmse(r_final, len(t_obs)),
                             k + 1, True, history, "levenberg-marquardt")

    r_final = _residuals(theta, t_obs, x_obs, y_obs, builder)
    return FitResult(_vec_to_params(theta, builder), _rmse(r_final, len(t_obs)),
                     max_iter, False, history, "levenberg-marquardt")


def gradient_descent(
    t_obs, x_obs, y_obs,
    theta0,
    builder,
    tol        = 1e-6,
    max_iter   = 500,
    alpha0     = 1.0,
    c_armijo   = 1e-4,
    beta       = 0.5,
    max_ls     = 30,
):
    theta   = np.array(theta0, dtype=float)
    history = []

    for k in range(max_iter):
        r    = _residuals(theta, t_obs, x_obs, y_obs, builder)
        cost = _cost(r)
        history.append(cost)

        J    = _jacobian_fd(theta, t_obs, x_obs, y_obs, builder)
        grad = J.T @ r
        gnorm = np.linalg.norm(grad)

        if gnorm < tol:
            return FitResult(_vec_to_params(theta, builder), _rmse(r, len(t_obs)),
                             k + 1, True, history, "gradient-descent")

        alpha = alpha0
        for _ in range(max_ls):
            theta_trial = theta - alpha * grad
            r_trial     = _residuals(theta_trial, t_obs, x_obs, y_obs, builder)
            if _cost(r_trial) <= cost - c_armijo * alpha * gnorm ** 2:
                break
            alpha *= beta

        theta = theta - alpha * grad

        if np.linalg.norm(alpha * grad) < tol:
            r_final = _residuals(theta, t_obs, x_obs, y_obs, builder)
            return FitResult(_vec_to_params(theta, builder), _rmse(r_final, len(t_obs)),
                             k + 1, True, history, "gradient-descent")

    r_final = _residuals(theta, t_obs, x_obs, y_obs, builder)
    return FitResult(_vec_to_params(theta, builder), _rmse(r_final, len(t_obs)),
                     max_iter, False, history, "gradient-descent")



def fit_trajectory(t_obs, x_obs, y_obs, model      = "vacuum", theta0     = None, method     = "lm", **kwargs,):
    if model not in _BUILDERS:
        raise ValueError(f"model must be 'vacuum' or 'drag', got '{model}'")
    builder, n_params = _BUILDERS[model]

    t_obs = np.asarray(t_obs, dtype=float)
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)

    if theta0 is None:
        theta0 = _naive_guess(t_obs, x_obs, y_obs, n_params)
    else:
        theta0 = np.asarray(theta0, dtype=float)

    if method == "lm":
        return levenberg_marquardt(t_obs, x_obs, y_obs, theta0, builder, **kwargs)
    elif method == "gd":
        return gradient_descent(t_obs, x_obs, y_obs, theta0, builder, **kwargs)
    else:
        raise ValueError(f"method must be 'lm' or 'gd', got '{method}'")


def compare_models(t_obs, x_obs, y_obs, **kwargs):
    res_vacuum = fit_trajectory(t_obs, x_obs, y_obs, model="vacuum", **kwargs)
    res_drag   = fit_trajectory(t_obs, x_obs, y_obs, model="drag",   **kwargs)
    return res_vacuum, res_drag



def _naive_guess(t, x, y, n_params):
    x0  = x[0]
    y0  = y[0]
    dt  = t[1] - t[0] if len(t) > 1 else 1 / 30
    vx0 = (x[1] - x[0]) / dt if len(x) > 1 else 0.0
    vy0 = (y[1] - y[0]) / dt if len(y) > 1 else 0.0
    theta = [x0, y0, vx0, vy0]
    if n_params == 5:
        theta.append(0.01)   # small initial drag
    return np.array(theta, dtype=float)

def _vec_to_params(theta, builder):
    try:
        return builder(theta).params()
    except Exception:
        return {"theta": theta.tolist()}

def _rmse(r, n_obs):
    return float(np.sqrt(np.mean(r ** 2)))