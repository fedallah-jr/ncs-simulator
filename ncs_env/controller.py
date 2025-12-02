from __future__ import annotations

from collections import deque
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
        max_delay: int = 100,
    ):
        self.A = A
        self.B = B
        self.K = K
        self.last_u = np.zeros(B.shape[1])
        self.current_state_index = 0

        state_dim = A.shape[0]
        if initial_covariance is None:
            self.initial_covariance = np.eye(state_dim)
        else:
            self.initial_covariance = initial_covariance.copy()

        # History storage for handling delayed measurements
        # Stores (timestep, x_prior, P_prior) before each update
        self.max_delay = max_delay
        self.prior_history: deque = deque(maxlen=max_delay)
        # Stores (timestep, u) for each control applied
        self.control_history: deque = deque(maxlen=max_delay)

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

    def store_prior(self, state_index: int):
        """
        Store current state and covariance as prior for this state index.

        Parameters
        ----------
        state_index : int
            The time index k of the state x[k] that will be measured.
            The stored prior x̂[k|k-1] is for estimating x[k].
        """
        self.prior_history.append({
            'state_index': state_index,
            'x': self.kf.x.copy(),
            'P': self.kf.P.copy()
        })
        self.current_state_index = state_index

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

    def delayed_update(self, measurement: np.ndarray, measurement_state_index: int) -> bool:
        """
        Handle a delayed measurement by retrodict-then-predict.

        1. Find the stored prior from when measurement was taken
        2. Apply Kalman update to get posterior for that time
        3. Re-predict forward to current time using stored controls

        Parameters
        ----------
        measurement : np.ndarray
            The measurement vector (measurement of x[measurement_state_index])
        measurement_state_index : int
            The state index k when the measurement was taken (i.e., measurement is of x[k])

        Returns
        -------
        bool
            True if update was successful, False if prior not found
        """
        # Find the prior for the measurement state index
        prior_entry = None
        for entry in self.prior_history:
            if entry['state_index'] == measurement_state_index:
                prior_entry = entry
                break

        if prior_entry is None:
            # Prior not found (too old or not stored); treat as stale and skip update
            return False

        # Restore the prior state
        x_prior = prior_entry['x'].copy()
        P_prior = prior_entry['P'].copy()
        # Apply Kalman update equations manually
        # y = z - H @ x_prior (innovation)
        H = self.kf.H
        R = self.kf.R
        z = measurement.reshape(-1, 1)

        y = z - H @ x_prior
        S = H @ P_prior @ H.T + R
        K_gain = P_prior @ H.T @ np.linalg.inv(S)

        x_posterior = x_prior + K_gain @ y
        P_posterior = (np.eye(P_prior.shape[0]) - K_gain @ H) @ P_prior
        # Collect controls from measurement_state_index to current_state_index
        controls = []
        for entry in self.control_history:
            if entry['state_index'] >= measurement_state_index and entry['state_index'] < self.current_state_index:
                controls.append(entry)

        # Sort by state_index to ensure correct order
        controls.sort(key=lambda x: x['state_index'])

        # Re-predict forward using the controls
        x_current = x_posterior
        P_current = P_posterior

        for ctrl_entry in controls:
            u = ctrl_entry['u'].reshape(-1, 1)
            # Predict: x = A @ x + B @ u, P = A @ P @ A.T + Q
            x_current = self.A @ x_current + self.B @ u
            P_current = self.A @ P_current @ self.A.T + self.kf.Q

        # Update the Kalman filter state
        self.kf.x = x_current
        self.kf.P = P_current
        return True

    def compute_control(self) -> np.ndarray:
        """Compute control input based on current estimate."""
        u = -self.K @ self.kf.x.flatten()
        self.last_u = u.copy()

        # Store control for delayed measurement handling
        # Control u[k] is computed at state index k and applied to transition x[k] -> x[k+1]
        self.control_history.append({
            'state_index': self.current_state_index,
            'u': u.copy()
        })

        return u

    def reset(self, initial_estimate: np.ndarray):
        """Reset controller state and covariance."""
        self.kf.x = initial_estimate.copy().reshape(-1, 1)
        self.last_u = np.zeros_like(self.last_u)
        self.kf.P = self.initial_covariance.copy()
        self.current_state_index = 0
        self.prior_history.clear()
        self.control_history.clear()
