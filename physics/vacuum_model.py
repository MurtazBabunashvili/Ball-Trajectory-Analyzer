import numpy as np


class VacuumTrajectory:
    def __init__(self, x0: float, y0: float,
                 vx0: float, vy0: float,
                 g: float = 9.81):
        self.x0  = float(x0)
        self.y0  = float(y0)
        self.vx0 = float(vx0)
        self.vy0 = float(vy0)
        self.g   = float(g)

    @classmethod
    def from_speed_angle(cls, speed: float, angle_deg: float,
                         x0: float = 0.0, y0: float = 0.0,
                         g: float = 9.81) -> "VacuumTrajectory":
        angle_rad = np.radians(angle_deg)
        vx0 = speed * np.cos(angle_rad)
        vy0 = speed * np.sin(angle_rad)
        return cls(x0, y0, vx0, vy0, g)

    def x(self, t):
        t = np.asarray(t, dtype=float)
        return self.x0 + self.vx0 * t

    def y(self, t):
        t = np.asarray(t, dtype=float)
        return self.y0 + self.vy0 * t - 0.5 * self.g * t ** 2

    def vx(self, t):
        t = np.asarray(t, dtype=float)
        return np.full_like(t, self.vx0)

    def vy(self, t):
        t = np.asarray(t, dtype=float)
        return self.vy0 - self.g * t

    def ax(self, t):
        t = np.asarray(t, dtype=float)
        return np.zeros_like(t)

    def ay(self, t):
        t = np.asarray(t, dtype=float)
        return np.full_like(t, -self.g)

    def speed(self, t):
        return np.sqrt(self.vx(t) ** 2 + self.vy(t) ** 2)

    def evaluate(self, t):
        t = np.asarray(t, dtype=float)
        return self.x(t), self.y(t)

    @property
    def t_apex(self) -> float:
        if self.g == 0:
            return None
        t = self.vy0 / self.g
        return t if t >= 0 else None

    @property
    def y_apex(self) -> float:
        ta = self.t_apex
        if ta is None:
            return None
        return float(self.y(ta))

    @property
    def t_land(self) -> float:
        if self.g == 0 or self.vy0 <= 0:
            return None
        return 2.0 * self.vy0 / self.g

    @property
    def range_(self) -> float:
        tl = self.t_land
        if tl is None:
            return None
        return float(self.x(tl) - self.x0)

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
        }

    def __repr__(self) -> str:
        return (
            f"VacuumTrajectory("
            f"x0={self.x0:.1f}, y0={self.y0:.1f}, "
            f"vx0={self.vx0:.2f}, vy0={self.vy0:.2f}, "
            f"g={self.g:.4f})"
        )