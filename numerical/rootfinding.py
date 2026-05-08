#Implemented methods:
# 1. Bisection
# 2. Newton-Raphson
# 3. Secant
# 4. Regula Falsi
# 5. Fixed-point iteration
# 6. Brent's algorithm
# 7. Aitken's acceleration
# 8. Steffensen's method

import numpy as np


#This class instance holds result of any rootfinding method used
class RootResult:

    def __init__(self, root, iterations, converged, history):
        self.root = root
        self.iterations = iterations
        self.converged = converged
        self.history = history

    def __repr__(self):
        status = "converged" if self.converged else "did not converge"
        return (f"RootResult(root={self.root:.10f}, "
                f"iters={self.iterations}, {status})")


#Bisection method
def bisection(f, a, b, tol=1e-8, max_iters=100):
    if f(a) * f(b) > 0:
        raise ValueError("Opposite signs requirement not met")

    history = []
    for i in range(1, max_iters + 1):
        m = (a+b)/2
        history.append(m)

        if abs(f(m)) < tol or (b-a)/2 < tol:
            return RootResult(m, i, True, history) #Converged, return

        if f(a) * f(m) < 0:
            b = m
        else:
            a = m
    return RootResult((a+b) /2, max_iters, False, history) #Did not converge


#Newton-raphson
def newton_raphson(f, df, x0, tol=1e-8, max_iter=100):
    x = x0
    history = [x]

    for i in range(1, max_iter+1):
        fx = f(x)
        if abs(fx) < tol:
            return RootResult(x, i, True, history) #Converged, return

        dfx = df(x)
        if abs(dfx) < 1e-14: #Not smooth near root
            raise ZeroDivisionError(f"Derivative zero at x={x}. Newton's method failed")

        x = x - fx/dfx
        history.append(x)

        if abs(history[-1] - history[-2]) < tol:
            return RootResult(x, i, True, history) #Converged, return

    return RootResult(x, max_iter, False, history) #Did not converge

#Secant method
def secant(f, x0, x1, tol=1e-8, max_iter=100):
    history = [x0, x1]

    for i in range(2, max_iter + 2):
        f0, f1 = f(x0), f(x1)
        if abs(f1 - f0) < 1e-14: #Too small to converge
            raise ZeroDivisionError("Secant denominator near zero")

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        history.append(x2)

        if abs(x2 - x1) < tol or abs(f(x2)) < tol:
            return RootResult(x2, i, True, history) #Converged

        x0, x1 = x1, x2
    return RootResult(x1, max_iter, False, history) #Did not converge


#Regula falsi
def regula_falsi(f, a, b, tol=1e-8, max_iter=100):
    if f(a) * f(b) > 0:
        raise ValueError("Opposite signs requirement not met")

    history = []

    for i in range(1, max_iter+1):
        fa, fb = f(a), f(b)
        c= b - fb * (b-a) / (fb - fa)
        history.append(c)
        fc = f(c)

        if abs(fc) < tol or abs(b-a) < tol: #Converged
            return RootResult(c, i, True, history)

        if fa * fc < 0:
            b = c
        else:
            a = c
    return RootResult(c, max_iter, False, history) #Did not converge

#Fixed-point iteration
def fixed_point(g, x0, tol=1e-8, max_iter=100):
    x= x0
    history = [x]

    for i in range(1, max_iter + 1):
        x_new = g(x)
        history.append(x_new)

        if abs(x_new - x) < tol: #Converged
            return RootResult(x_new, i, True, history)

        x = x_new
    return RootResult(x, max_iter, False, history) #Did not converge

#Brent's algorithm

def brent(f, a, b, tol=1e-8, max_iter=100):
    if f(a) * f(b) > 0:
        raise ValueError("Opposite signs requirement not met")

    if abs(f(a)) < abs(f(b)):
        a, b = b, a

    c = a
    mflag = True
    s = 0.0
    d = 0.0
    history = []

    for i in range(1, max_iter + 1):
        fa, fb, fc = f(a), f(b), f(c)

        if fa != fc and fb != fc: #Inverse quadratic interpolation
            s = (a * fb * fc / ((fa - fb) * (fa - fc)) +
                 b * fa * fc / ((fb - fa) * (fb - fc)) +
                 c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            s = b - fb * (b - a) / (fb - fa)   # secant

        cond1 = not ((3*a + b) / 4 < s < b or b < s < (3*a + b) / 4)
        cond2 = mflag  and abs(s - b) >= abs(b - c) / 2
        cond3 = not mflag and abs(s - b) >= abs(c - d) / 2

        if cond1 or cond2 or cond3:
            s = (a+b) / 2 #Go back to bisection
            mflag = True
        else:
            mflag = False

        history.append(s)
        fs = f(s)
        d, c = c, b
        if fa * fs < 0:
            b = s
        else:
            a = s

        if abs(f(a)) < abs(f(b)):
            a, b = b, a

        if abs(fb) < tol or abs(b-a) <tol:
            return RootResult(b, i, True, history) #Converged

    return RootResult(b, max_iter, False, history)#Did not converge

#Aitken's acceleration
def aitken_acceleration(sequence):
    p = np.asarray(sequence, dtype=float)
    num = (p[1:-1] - p[:-2]) ** 2
    den = p[2:] - 2 * p[1:-1] + p[:-2]

    safe = np.abs(den) > 1e-14
    acc = np.where(safe, p[:-2] - num / np.where(safe, den, 1.0), p[:-2])
    return acc

#Steffensen's method
def steffensen(g, x0, tol=1e-8, max_iter=100):
    x = x0
    history = [x]

    for i in range(1, max_iter + 1):
        p1 = g(x)
        p2 = g(p1)
        den = p2 - 2 * p1 + x

        if abs(den) < 1e-14:
            return RootResult(x, i, False, history)  # Diverged

        x_new = x - (p1 - x) ** 2 / den
        history.append(x_new)

        if abs(x_new - x) < tol:
            return RootResult(x_new, i, True, history) #converged

        x = x_new

    return RootResult(x, max_iter, False, history) #Did not converge


#Trajectory interface
def find_apex(vy_func, t_start, t_end, tol=1e-8):
    return brent(vy_func, t_start, t_end, tol=tol)

def find_landing(y_func, t_launch, t_max, tol=1e-8):
    return brent(y_func, t_launch, t_max, tol=tol)

def find_launch_angle_for_target(range_func, angle_min, angle_max, tol=1e-8):
    return brent(range_func, angle_min, angle_max, tol=tol)

