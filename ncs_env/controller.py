from __future__ import annotations

from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.linalg import solve_discrete_are


def compute_discrete_lqr_gain(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """
    Compute the optimal LQR gain matrix using scipy's DARE solver.

    Solves the discrete-time algebraic Riccati equation (DARE) and computes
    the optimal feedback gain:
        K = (R + B^T P B)^{-1} (B^T P A)

    Parameters
    ----------
    A : np.ndarray
        State transition matrix
    B : np.ndarray
        Control input matrix
    Q : np.ndarray
        State cost matrix
    R : np.ndarray
        Control cost matrix

    Returns
    -------
    K : np.ndarray
        Optimal LQR feedback gain matrix
    """
    # Solve discrete-time algebraic Riccati equation using scipy
    P = solve_discrete_are(A, B, Q, R)

    # Compute optimal gain from solution
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

    return K


class Controller:
    """
    Remote controller with Kalman-filtered state estimation and LQR control.
    Uses u[k] = -K * x_hat[k].

    Now uses filterpy's KalmanFilter for proper predict-then-update cycle.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        K: np.ndarray,
        initial_estimate: np.ndarray,
        process_noise_cov: np.ndarray,
        measurement_noise_cov: np.ndarray,
        initial_covariance: Optional[np.ndarray] = None,
    ):
        self.A = A
        self.B = B
        self.K = K
        self.last_u = np.zeros(B.shape[1])

        state_dim = A.shape[0]
        if initial_covariance is None:
            self.initial_covariance = np.eye(state_dim)
        else:
            self.initial_covariance = initial_covariance.copy()

        # Create filterpy KalmanFilter
        # dim_x: state dimension, dim_z: measurement dimension
        self.kf = KalmanFilter(dim_x=state_dim, dim_z=state_dim)

        # State transition matrix
        self.kf.F = A.copy()

        # Control input matrix
        self.kf.B = B.copy()

        # Measurement function (identity - we measure full state)
        self.kf.H = np.eye(state_dim)

        # Process noise covariance
        self.kf.Q = process_noise_cov.copy()

        # Measurement noise covariance
        self.kf.R = measurement_noise_cov.copy()

        # Initial state estimate
        self.kf.x = initial_estimate.copy().reshape(-1, 1)

        # Initial covariance
        self.kf.P = self.initial_covariance.copy()

    @property
    def x_hat(self) -> np.ndarray:
        """Get current state estimate."""
        return self.kf.x.flatten()

    @property
    def P(self) -> np.ndarray:
        """Get current covariance matrix."""
        return self.kf.P

    def predict(self):
        """
        Predict the next state estimate and covariance.
        This is the time update step: propagates state forward using dynamics and control.
        """
        # filterpy's predict can take control input
        self.kf.predict(u=self.last_u.reshape(-1, 1))

    def update(self, measurement: np.ndarray):
        """
        Kalman filter measurement update.
        This is the measurement update step: corrects prediction using measurement.
        """
        self.kf.update(measurement.reshape(-1, 1))

    def compute_control(self) -> np.ndarray:
        """Compute control input based on current estimate."""
        u = -self.K @ self.kf.x.flatten()
        self.last_u = u.copy()
        return u

    def reset(self, initial_estimate: np.ndarray):
        """Reset controller state and covariance."""
        self.kf.x = initial_estimate.copy().reshape(-1, 1)
        self.last_u = np.zeros_like(self.last_u)
        self.kf.P = self.initial_covariance.copy()
