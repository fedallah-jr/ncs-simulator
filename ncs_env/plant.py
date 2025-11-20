import numpy as np


class Plant:
    """
    Physical plant model for a single control loop.
    Implements: x[k+1] = A*x[k] + B*u[k] + w[k] with Gaussian process noise.
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, W: np.ndarray, x0: np.ndarray, rng: np.random.Generator = None):
        self.A = A
        self.B = B
        self.W = W
        self.x = x0.copy()
        self.state_dim = A.shape[0]
        # Use provided RNG or create a new isolated one
        self.rng = rng if rng is not None else np.random.default_rng()

    def step(self, u: np.ndarray) -> np.ndarray:
        """Update plant state for one timestep."""
        w = self.rng.multivariate_normal(np.zeros(self.state_dim), self.W)
        self.x = self.A @ self.x + self.B @ u + w
        return self.x.copy()

    def get_state(self) -> np.ndarray:
        """Return the current state."""
        return self.x.copy()

    def reset(self, x0: np.ndarray):
        """Reset plant to a provided initial state."""
        self.x = x0.copy()
