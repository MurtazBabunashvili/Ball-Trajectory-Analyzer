#Implemented methods:
#1. Lagrange interpolation
#2. Newton's divided differences
#3. Hermite interpolation via divided differences
#4. Cubic spline interpolation (natural and clamped)

import numpy as np

#Lagrange interpolation
def lagrange_basis(x, nodes, k):
    nodes = np.asarray(nodes, dtype=float)
    result = 1.0
    xk = nodes[k]
    for j, xj in enumerate(nodes):
        if j != k:
            result *= (x-xj)/(xk-xj)
    return result

def lagrange_interpolate(x, nodes, values):
    nodes = np.asarray(nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    x = np.asarray(x, dtype=float)

    scalar = x.ndim == 0
    x = np.atleast_1d(x)

    result = np.zeros_like(x)
    for k, yk in enumerate(values):
        basis = np.array([lagrange_basis(xi, nodes, k) for xi in x])
        result += yk * basis
    return result[0] if scalar else result

#Newton's divided difference
def divided_difference_table(nodes, values):
    nodes = np.asarray(nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(nodes)
    table = np.zeros((n,n))

    table[:, 0] = values #Zeroth order

    for j in range(1, n):
        for i in range(n-j):
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (nodes[i+j] - nodes[i])

    return table

def newton_interpolate(x, nodes, values):
    nodes = np.asarray(nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)

    table = divided_difference_table(nodes, values)
    coefficients = table[0, :]

    result = np.zeros_like(x)
    for xi_idx, xi in enumerate(x):
        total   = 0.0
        product = 1.0
        for k, ck in enumerate(coefficients):
            total   += ck * product
            product *= (xi - nodes[k])
        result[xi_idx] = total

    return result[0] if scalar else result

#Hermite interpolation
def hermite_interpolate(x, nodes, values, derivatives):
    nodes = np.asarray(nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    derivatives = np.asarray(derivatives, dtype=float)
    n = len(nodes)

    m = 2*n
    z = np.zeros(m)
    fz = np.zeros(m)
    for i in range(n):
        z[2*i] = nodes[i]
        z[2*i + 1] = nodes[i]
        fz[2*i] = values[i]
        fz[2*i + 1] = values[i]

    #Divided difference table
    Q = np.zeros((m, m))
    Q[:, 0] = fz #Zeroth order

    for j in range(1, m):
        for i in range(m-j):
            if np.isclose(z[i+j], z[i]): #0/0 form
                Q[i][j] = derivatives[i//2]
            else:
                Q[i][j] = (Q[i+1][j-1] - Q[i][j-1])/(z[i+j] - z[i])


    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)

    result = np.zeros_like(x)
    for xi_idx, xi in enumerate(x):
        total = 0.0
        product = 1.0
        for k in range(m):
            total += Q[0][k] * product
            product *= (xi - z[k])
        result[xi_idx] = total

    return result[0] if scalar else result


#Cubic spline interpolation

def solve_tridiagonal(lower, main, upper, rhs):
    #This method uses thomas algorithm from NLA course
    n     = len(main)
    c     = np.array(upper, dtype=float)
    d     = np.array(rhs,   dtype=float)
    a     = np.array(lower, dtype=float)
    b     = np.array(main,  dtype=float)

    for i in range(1, n):
        w = a[i-1]/b[i-1]
        b[i] = b[i] - w*c[i-1]
        d[i] = d[i] - w*d[i-1]

    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]

    return x

def cubic_spline(nodes, values, mode="natural", df_left=None, df_right = None):
    nodes = np.asarray(nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(nodes) - 1
    h = np.diff(nodes)

    rhs = 6 * ((values[2:] - values[1:-1]) / h[1:] -
               (values[1:-1] - values[:-2]) / h[:-1])

    if mode == "natural":
        main = 2 * (h[:-1] + h[1:])
        upper = h[1:-1]
        lower = h[1:-1]
        M_interior = solve_tridiagonal(lower, main, upper, rhs)
        M = np.concatenate([[0.0], M_interior, [0.0]])

    elif mode == "clamped":
        if df_left is None or df_right is None:
            raise ValueError("Spline requires df_left and df_right.")

        size = n + 1
        main = np.zeros(size)
        upper = np.zeros(size)
        lower = np.zeros(size)
        rhs_c = np.zeros(size)

        main[0] = 2 * h[0]
        upper[0] = h[0]
        rhs_c[0] = 6 * ((values[1] - values[0]) / h[0] - df_left)

        for i in range(1, n):
            lower[i] = h[i - 1]
            main[i] = 2 * (h[i - 1] + h[i])
            upper[i] = h[i]
            rhs_c[i] = 6 * ((values[i + 1] - values[i]) / h[i] -
                            (values[i] - values[i - 1]) / h[i - 1])

        lower[n] = h[n - 1]
        main[n] = 2 * h[n - 1]
        rhs_c[n] = 6 * (df_right - (values[n] - values[n - 1]) / h[n - 1])

        M = solve_tridiagonal(lower[1:], main, upper[:-1], rhs_c)

    else:
        raise ValueError("mode must be 'natural' or 'clamped'.")

    return {"nodes": nodes, "values": values, "M": M, "h": h}


def cubic_spline_evaluate(spline_data, x):
    nodes = spline_data["nodes"]
    values = spline_data["values"]
    M = spline_data["M"]
    h = spline_data["h"]

    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    result = np.zeros_like(x)

    for idx, xi in enumerate(x):
        i = np.searchsorted(nodes, xi, side="right") - 1
        i = np.clip(i, 0, len(h) - 1)

        hi = h[i]
        dx_right = nodes[i + 1] - xi
        dx_left = xi - nodes[i]

        result[idx] = (
                (M[i] / (6 * hi)) * dx_right ** 3 +
                (M[i + 1] / (6 * hi)) * dx_left ** 3 +
                (values[i] / hi - M[i] * hi / 6) * dx_right +
                (values[i + 1] / hi - M[i + 1] * hi / 6) * dx_left
        )

    return result[0] if scalar else result


#Trajectory interface. given (t,x,y positions we return (x(t), y(t)) curves (smooth)
def interpolate_trajectory(t_known, x_known, y_known, t_query, method="spline", derivatives=None):
    method = method.lower()

    if method == "lagrange":
        x_interp = lagrange_interpolate(t_query, t_known, x_known)
        y_interp = lagrange_interpolate(t_query, t_known, y_known)

    elif method == "newton":
        x_interp = newton_interpolate(t_query, t_known, x_known)
        y_interp = newton_interpolate(t_query, t_known, y_known)

    elif method == "hermite":
        if derivatives is None:
            raise ValueError("Hermite requires derivatives={'vx': ..., 'vy': ...}")
        x_interp = hermite_interpolate(t_query, t_known, x_known, derivatives["vx"])
        y_interp = hermite_interpolate(t_query, t_known, y_known, derivatives["vy"])

    elif method == "spline":
        sx = cubic_spline(t_known, x_known, mode="natural")
        sy = cubic_spline(t_known, y_known, mode="natural")
        x_interp = cubic_spline_evaluate(sx, t_query)
        y_interp = cubic_spline_evaluate(sy, t_query)

    else:
        raise ValueError(f"Unknown method: {method}. "
                         "Choose: lagrange | newton | hermite | spline")

    return x_interp, y_interp