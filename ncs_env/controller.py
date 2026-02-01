from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Set, Tuple, Union

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.linalg import solve_discrete_are


def compute_discrete_lqr_solution(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the optimal LQR gain matrix and cost-to-go matrix using scipy's DARE solver.

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
    P : np.ndarray
        Optimal cost-to-go (Riccati) matrix
    """
    # Solve discrete-time algebraic Riccati equation using scipy
    P = solve_discrete_are(A, B, Q, R)

    # Compute optimal gain from solution
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

    return K, P


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
    K, _ = compute_discrete_lqr_solution(A, B, Q, R)
    return K


def compute_finite_horizon_lqr_solution(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    horizon: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Compute finite-horizon LQR gains and cost-to-go matrices using backward dynamic Riccati recursion.

    Solves the discrete-time dynamic Riccati equation (DRE) backward from
    terminal time k=N to k=0:

        P[N] = Q

        For k = N-1, N-2, ..., 0:
            S[k] = R + B^T P[k+1] B
            K[k] = S[k]^{-1} B^T P[k+1] A
            P[k] = Q + A^T P[k+1] A - A^T P[k+1] B K[k]

    This is the standard discrete-time LQR backward recursion for optimal
    control with finite horizon. See Bertsekas "Dynamic Programming and
    Optimal Control".

    Parameters
    ----------
    A : np.ndarray
        State transition matrix (n × n)
    B : np.ndarray
        Control input matrix (n × m)
    Q : np.ndarray
        State cost matrix (n × n)
    R : np.ndarray
        Control cost matrix (m × m)
    horizon : int
        Episode length (number of timesteps)

    Returns
    -------
    gains : List[np.ndarray]
        List of time-varying gains [K[0], K[1], ..., K[N-1]]
        where K[k] is the optimal gain at timestep k (m × n matrix)
    costs : List[np.ndarray]
        List of cost-to-go matrices [P[0], P[1], ..., P[N-1]]
    """
    # Terminal cost
    P_next = Q.copy()

    # Store gains in reverse order (computed backward)
    gains: List[np.ndarray] = []
    costs: List[np.ndarray] = []

    # Backward recursion from k=N-1 down to k=0
    for k in range(horizon - 1, -1, -1):
        # Compute gain at timestep k
        S = R + B.T @ P_next @ B
        K = np.linalg.solve(S, B.T @ P_next @ A)  # More stable than inv(S) @ ...

        # Update P for previous timestep
        P = Q + A.T @ P_next @ A - A.T @ P_next @ B @ K
        gains.append(K)
        costs.append(P)
        P_next = P

    # Reverse to get forward-time order [K[0], K[1], ..., K[N-1]]
    gains.reverse()
    costs.reverse()

    return gains, costs


def compute_finite_horizon_lqr_gains(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    horizon: int,
) -> List[np.ndarray]:
    """
    Compute finite-horizon LQR gains using backward dynamic Riccati recursion.

    Solves the discrete-time dynamic Riccati equation (DRE) backward from
    terminal time k=N to k=0.
    """
    gains, _ = compute_finite_horizon_lqr_solution(A, B, Q, R, horizon)
    return gains


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
        K: Union[np.ndarray, List[np.ndarray]],
        initial_estimate: np.ndarray,
        process_noise_cov: np.ndarray,
        measurement_noise_cov: np.ndarray,
        initial_covariance: Optional[np.ndarray] = None,
        max_delay: int = 100,
    ):
        self.A = A
        self.B = B

        # Support both static and time-varying gains
        self.use_finite_horizon = isinstance(K, list)
        if self.use_finite_horizon:
            self.K_list: List[np.ndarray] = K
            self.K = K[0]  # For compatibility with existing code that checks self.K
        else:
            self.K: np.ndarray = K
            self.K_list = None

        # Time tracking for finite-horizon mode
        self.time_step: int = 0

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
        # Track measurement indices already processed to avoid duplicate updates.
        self.seen_measurement_indices: Deque[int] = deque()
        self.seen_measurement_set: Set[int] = set()

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

    def update(self, measurement: np.ndarray, measurement_noise_cov: Optional[np.ndarray] = None) -> None:
        """
        Kalman filter measurement update.
        This is the measurement update step: corrects prediction using measurement.
        """
        if measurement_noise_cov is None:
            self.kf.update(measurement.reshape(-1, 1))
        else:
            self.kf.update(measurement.reshape(-1, 1), R=measurement_noise_cov)
        self._record_seen_measurement(self.current_state_index)

    def _has_seen_measurement(self, measurement_state_index: int) -> bool:
        return measurement_state_index in self.seen_measurement_set

    def _record_seen_measurement(self, measurement_state_index: int) -> None:
        if measurement_state_index in self.seen_measurement_set:
            return
        self.seen_measurement_indices.append(measurement_state_index)
        self.seen_measurement_set.add(measurement_state_index)
        while len(self.seen_measurement_indices) > self.max_delay:
            old_index = self.seen_measurement_indices.popleft()
            self.seen_measurement_set.discard(old_index)

    def delayed_update(
        self,
        measurement: np.ndarray,
        measurement_state_index: int,
        measurement_noise_cov: Optional[np.ndarray] = None,
    ) -> bool:
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
        measurement_state_index = int(measurement_state_index)
        if self._has_seen_measurement(measurement_state_index):
            return False
        # Find the prior for the measurement state index
        prior_entry = None
        for entry in self.prior_history:
            if entry['state_index'] == measurement_state_index:
                prior_entry = entry
                break

        if prior_entry is None:
            # Prior not found (too old or not stored); treat as stale and skip update
            self._record_seen_measurement(measurement_state_index)
            return False

        # Restore the prior state
        x_prior = prior_entry['x'].copy()
        P_prior = prior_entry['P'].copy()
        # Apply Kalman update equations manually
        # y = z - H @ x_prior (innovation)
        H = self.kf.H
        R = measurement_noise_cov if measurement_noise_cov is not None else self.kf.R
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
        self._record_seen_measurement(measurement_state_index)
        return True

    def compute_control(self) -> np.ndarray:
        """Compute control input based on current estimate."""
        if self.use_finite_horizon:
            # Use time-varying gain K[k], clamped to last gain if episode exceeds pre-computed horizon
            k = min(self.time_step, len(self.K_list) - 1)
            K_now = self.K_list[k]
            u = -K_now @ self.kf.x.flatten()
            self.time_step += 1
        else:
            # Original infinite-horizon static gain
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
        self.seen_measurement_indices.clear()
        self.seen_measurement_set.clear()

        # Reset time counter for finite-horizon mode
        self.time_step = 0
