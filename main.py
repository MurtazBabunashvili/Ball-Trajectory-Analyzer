import math


# =========================================================
# Dynamic-bound Brent search
#
# Left bound:
#     a(t) -> -infinity
#
# Right bound:
#     b(t) -> +infinity
#
# Example:
#     a(t) = -t
#     b(t) =  t
#
# We expand interval automatically until:
#
#     f(a(t)) * f(b(t)) < 0
#
# then run Brent.
# =========================================================


def brent(f, a, b, eps=1e-8, max_iters=1000):

    fa = f(a)
    fb = f(b)

    if fa * fb >= 0:
        return None

    for _ in range(max_iters):

        # secant step
        if fb != fa:
            x = b - fb * (b - a) / (fb - fa)
        else:
            x = (a + b) / 2

        # fallback to bisection
        if x < min(a, b) or x > max(a, b):
            x = (a + b) / 2

        fx = f(x)

        # interval update
        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx

        # convergence
        if abs(b - a) < eps:
            return x

    return x


# =========================================================
# YOUR IDEA:
#
# bounds are FUNCTIONS approaching infinities
# =========================================================

def global_brent(
        f,
        left_bound_func,
        right_bound_func,
        search_iters=1000,
        eps=1e-8):

    for t in range(1, search_iters + 1):

        a = left_bound_func(t)
        b = right_bound_func(t)

        fa = f(a)
        fb = f(b)

        print(f"t={t}, interval=[{a}, {b}]")

        # sign change found
        if fa * fb < 0:

            print("Bracket found!")

            root = brent(f, a, b, eps)

            # =====================================
            # integer projection idea
            # =====================================

            nearest = round(root)

            if f(nearest) == 0:
                print("Exact integer root:", nearest)

            return root

    return None


# =========================================================
# Example polynomial
# =========================================================
def f(x):
    return x**2 - 2


# =========================================================
# Dynamic bounds
#
# a(t) -> -infinity
# b(t) -> +infinity
# =========================================================

def a(t):
    return -t


def b(t):
    return t


root = global_brent(f, a, b)

print("\nApprox root:", root)