import numpy as np

_TENNIS = {
    "mass":    0.0574,   # kg   (ITF regulation)
    "radius":  0.0330,   # m
    "Cd":      0.507,    # dimensionless drag coefficient (sphere, Re~1e5)
    "rho":     1.204,    # kg/m³  air density at 20 °C, sea level
}


def _drag_coeff(mass, radius, Cd, rho) -> float:
    A = np.pi * radius ** 2
    return 0.5 * rho * Cd * A / mass   # k/m  [1/m]


class DragTrajectory:
    def __init__(
        self,
        x0:  float,
        y0:  float,
        vx0: float,
        vy0: float,
        g:   float = 9.81,
        km:  float = 0.0,
        dt_internal: float = 1 / 300,
        t_max:       float = 5.0,
    ):
        self.x0  = float(x0)
        self.y0  = float(y0)
        self.vx0 = float(vx0)
        self.vy0 = float(vy0)
        self.g   = float(g)
        self.km  = float(km)
        self.dt_internal = float(dt_internal)
        self.t_max       = float(t_max)

        self._t, self._x, self._y, self._vx, self._vy = self._integrate()


    @classmethod
    def from_speed_angle(
        cls,
        speed:     float,
        angle_deg: float,
        x0:  float = 0.0,
        y0:  float = 0.0,
        g:   float = 9.81,
        km:  float = 0.0,
        **kwargs,
    ) -> "DragTrajectory":
        angle_rad = np.radians(angle_deg)
        vx0 = speed * np.cos(angle_rad)
        vy0 = speed * np.sin(angle_rad)
        return cls(x0, y0, vx0, vy0, g, km, **kwargs)

    @classmethod
    def for_tennis_ball(
        cls,
        x0:  float,
        y0:  float,
        vx0: float,
        vy0: float,
        g:   float = 9.81,
        **kwargs,
    ) -> "DragTrajectory":
        km = _drag_coeff(
            _TENNIS["mass"],
            _TENNIS["radius"],
            _TENNIS["Cd"],
            _TENNIS["rho"],
        )
        return cls(x0, y0, vx0, vy0, g, km, **kwargs)

    # ── Forward Euler integrator ──────────────────────────────────────────────

    def _integrate(self):

        dt   = self.dt_internal
        n    = int(np.ceil(self.t_max / dt)) + 1

        t_arr  = np.zeros(n)
        x_arr  = np.zeros(n)
        y_arr  = np.zeros(n)
        vx_arr = np.zeros(n)
        vy_arr = np.zeros(n)

        # Initial conditions
        x_arr[0]  = self.x0
        y_arr[0]  = self.y0
        vx_arr[0] = self.vx0
        vy_arr[0] = self.vy0

        for i in range(n - 1):
            vx_i = vx_arr[i]
            vy_i = vy_arr[i]

            speed = np.sqrt(vx_i ** 2 + vy_i ** 2)

            ax = -self.km * speed * vx_i
            ay = -self.g  - self.km * speed * vy_i

            # Forward Euler step
            x_arr[i + 1]  = x_arr[i]  + dt * vx_i
            y_arr[i + 1]  = y_arr[i]  + dt * vy_i
            vx_arr[i + 1] = vx_i      + dt * ax
            vy_arr[i + 1] = vy_i      + dt * ay
            t_arr[i + 1]  = t_arr[i]  + dt

        return t_arr, x_arr, y_arr, vx_arr, vy_arr


    def _interp(self, t, arr) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        t_clipped = np.clip(t, self._t[0], self._t[-1])
        return np.interp(t_clipped, self._t, arr)

    def x(self, t):
        return self._interp(t, self._x)

    def y(self, t):
        return self._interp(t, self._y)

    def vx(self, t):
        return self._interp(t, self._vx)

    def vy(self, t):
        return self._interp(t, self._vy)

    def ax(self, t):
        t  = np.asarray(t, dtype=float)
        dt = self.dt_internal
        return (self.vx(t + dt) - self.vx(t - dt)) / (2 * dt)

    def ay(self, t):
        t  = np.asarray(t, dtype=float)
        dt = self.dt_internal
        return (self.vy(t + dt) - self.vy(t - dt)) / (2 * dt)

    def speed(self, t):
        return np.sqrt(self.vx(t) ** 2 + self.vy(t) ** 2)

    def evaluate(self, t):
        t = np.asarray(t, dtype=float)
        return self.x(t), self.y(t)


    @property
    def t_apex(self) -> float | None:
        vy = self._vy
        for i in range(len(vy) - 1):
            if vy[i] >= 0 > vy[i + 1]:
                t0, t1 = self._t[i], self._t[i + 1]
                v0, v1 = vy[i], vy[i + 1]
                return float(t0 - v0 * (t1 - t0) / (v1 - v0))
        return None

    @property
    def y_apex(self) -> float | None:
        ta = self.t_apex
        return None if ta is None else float(self.y(ta))

    @property
    def t_land(self) -> float | None:
        y0 = self.y0
        y  = self._y
        for i in range(1, len(y) - 1):
            if y[i] >= y0 > y[i + 1]:
                t0, t1 = self._t[i], self._t[i + 1]
                y_i, y_i1 = y[i] - y0, y[i + 1] - y0
                return float(t0 - y_i * (t1 - t0) / (y_i1 - y_i))
        return None

    @property
    def range_(self) -> float | None:
        tl = self.t_land
        return None if tl is None else float(self.x(tl) - self.x0)

    def residuals(self, t_obs, x_obs, y_obs) -> np.ndarray:
        t_obs = np.asarray(t_obs, dtype=float)
        x_obs = np.asarray(x_obs, dtype=float)
        y_obs = np.asarray(y_obs, dtype=float)
        x_pred, y_pred = self.evaluate(t_obs)
        return np.concatenate([x_pred - x_obs, y_pred - y_obs])

    def rmse(self, t_obs, x_obs, y_obs) -> float:
        r = self.residuals(t_obs, x_obs, y_obs)
        return float(np.sqrt(np.mean(r ** 2)))


    def params(self) -> dict:
        return {
            "x0":  self.x0,
            "y0":  self.y0,
            "vx0": self.vx0,
            "vy0": self.vy0,
            "g":   self.g,
            "km":  self.km,
        }

    def __repr__(self) -> str:
        return (
            f"DragTrajectory("
            f"x0={self.x0:.1f}, y0={self.y0:.1f}, "
            f"vx0={self.vx0:.2f}, vy0={self.vy0:.2f}, "
            f"g={self.g:.4f}, km={self.km:.6f})"
        )
