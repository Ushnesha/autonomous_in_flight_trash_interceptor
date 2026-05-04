"""
src/predict.py
Kalman filter + ballistic predictor.
Same interface as original predict.py:
    add_point((cx, cy))
    get_predicted_landing_x() → float (pixels) or None
    reset()

Fixes the original:
  - Kalman filter smooths noisy detections
  - Works from frame 2 (not frame 4+)
  - All math stays in world-meters, converts back to pixels at end
  - Uses FRAME_HEIGHT/ARENA_H for vertical scale (not ARENA_W)
"""

import math
import time
import numpy as np


class Predictor:

    def __init__(self, settings):
        self.S = settings
        self._history  = []      # list of {x, z, t} in world meters
        self._last_x   = None    # last predicted landing pixel X
        self._updates  = 0

        # Kalman state [x, z, vx, vz]
        self._kx  = None
        self._kP  = np.eye(4) * 1.0
        self._dt  = settings.SIM_TIMESTEP

        # pixels → meters conversion
        self._ppm_x = settings.FRAME_WIDTH  / settings.ARENA_W
        self._ppm_z = settings.FRAME_HEIGHT / settings.ARENA_H

        print(f"[PREDICT] Min points: {settings.MIN_POINTS_TO_PREDICT}")
        print(f"[PREDICT] ppm_x={self._ppm_x:.1f}  ppm_z={self._ppm_z:.1f}")

    def reset(self):
        self._history  = []
        self._last_x   = None
        self._updates  = 0
        self._kx       = None
        self._kP       = np.eye(4) * 1.0

    def add_point(self, position):
        """
        Feed a detector centroid.
        position can be (cx, cy) or (cx, cy, depth) — depth is ignored
        because the simulator is 2-D.
        """
        cx = position[0]
        cy = position[1]

        # Pixel → world metres
        # X: 0 = left edge → negative half-arena, centre = 0
        wx = (cx / self.S.FRAME_WIDTH  - 0.5) * self.S.ARENA_W

        # Z (height): pixel Y=0 is top of frame = highest point
        # pixel Y = FRAME_HEIGHT is floor level
        wz = (1.0 - cy / self.S.FRAME_HEIGHT) * self.S.ARENA_H

        self._kalman_update(wx, wz)
        self._updates += 1

    def get_predicted_landing_x(self):
        """
        Simplified Linear Prediction for real-world horizontal catching.
        """
        # Wait for enough points to get a stable velocity
        if self._updates < self.S.MIN_POINTS_TO_PREDICT or self._kx is None:
            return None

        # Extract current state from Kalman Filter
        # x0 = current horizontal pos, z0 = current distance/height
        # vx = horizontal velocity, vz = vertical/depth velocity
        x0, z0, vx, vz = self._kx.flatten()

        # 1. How much 'time' until the ball reaches the catch zone?
        # If the ball is moving down (negative vz), we estimate when it hits 'FLOOR_Y'
        if abs(vz) < 0.1: # Ball isn't really moving vertically
            return x0 

        # Time = Distance to floor / speed
        t_impact = (self.S.FLOOR_Y - z0) / vz

        # If t_impact is negative, the ball is moving AWAY from the catch zone
        if t_impact < 0:
            return x0 # Just track its current X

        # 2. Predict where X will be at that time
        landing_x_meters = x0 + (vx * t_impact)

        # 3. Convert meters back to pixels for the motor controller
        # X: (meters / arena_w + 0.5) * frame_width
        landing_x_px = (landing_x_meters / self.S.ARENA_W + 0.5) * self.S.FRAME_WIDTH

        # Clamp to screen bounds (0 to 320)
        landing_x_px = max(0, min(self.S.FRAME_WIDTH, landing_x_px))

        self._last_x = landing_x_px
        return landing_x_px

    def get_velocity(self):
        """Returns (vx, vz) in world-metres/s."""
        if self._kx is None:
            return (0.0, 0.0)
        return (float(self._kx[2, 0]), float(self._kx[3, 0]))

    # ── Kalman internals ─────────────────────────────────────────

    def _kalman_update(self, wx, wz):
        dt = self._dt
        g  = self.S.GRAVITY

        # State: [x, z, vx, vz]
        # Transition: constant-velocity + gravity on z
        #   x  ← x  + vx*dt
        #   z  ← z  + vz*dt - 0.5*g*dt²   (gravity pulls z down)
        #   vx ← vx
        #   vz ← vz - g*dt
        F = np.array([
            [1, 0, dt, 0 ],
            [0, 1, 0,  dt],
            [0, 0, 1,  0 ],
            [0, 0, 0,  1 ],
        ], dtype=float)

        # Gravity enters as a known control input, not noise
        u = np.array([[0.0],
                      [-0.5 * g * dt**2],
                      [0.0],
                      [-g * dt]])

        # Observation: we directly measure x and z
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Process noise — tune KALMAN_Q in sim_settings
        q = self.S.KALMAN_Q
        Q = np.diag([q * 0.1, q * 0.1, q, q])

        # Measurement noise — tune KALMAN_R in sim_settings
        r = self.S.KALMAN_R
        R = np.eye(2) * r

        # Measurement vector
        z_meas = np.array([[wx], [wz]])

        # ── Initialise on first observation ──────────────────────
        if self._kx is None:
            self._kx = np.array([[wx], [wz], [0.0], [0.0]])
            self._kP = np.eye(4) * 1.0
            return

        # ── Predict step ─────────────────────────────────────────
        xp = F @ self._kx + u
        Pp = F @ self._kP @ F.T + Q

        # ── Update step ──────────────────────────────────────────
        S_mat = H @ Pp @ H.T + R
        K     = Pp @ H.T @ np.linalg.inv(S_mat)
        y_err = z_meas - H @ xp
        self._kx = xp + K @ y_err
        self._kP = (np.eye(4) - K @ H) @ Pp















