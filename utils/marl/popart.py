"""PopArt (Preserving Outputs Precisely, while Adaptively Rescaling Targets).

Implements a linear output layer that maintains running statistics and corrects
its own weights/biases whenever statistics change, ensuring that the
*unnormalized* network output is preserved exactly across updates.

Reference:
    van Hasselt et al., "Learning values across many orders of magnitude", 2016.
"""

from __future__ import annotations

import torch
from torch import nn


class PopArtLayer(nn.Module):
    """Linear layer with PopArt output-preserving normalization.

    Maintains EMA statistics (mu, sigma) and corrects the layer's weight/bias
    whenever ``update_and_correct`` is called so that::

        sigma_old * (W_old @ h + b_old) + mu_old  ==  sigma_new * (W_new @ h + b_new) + mu_new

    for any hidden activation *h*.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        beta: float = 0.999,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.beta = beta
        self.eps = eps

        self.linear = nn.Linear(in_features, out_features)

        # EMA accumulators (not parameters – saved via register_buffer)
        self.register_buffer("_ema_mean", torch.zeros(out_features))
        self.register_buffer("_ema_mean_sq", torch.zeros(out_features))
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long))

        # Bias-corrected statistics exposed for normalize / denormalize
        self.register_buffer("mu", torch.zeros(out_features))
        self.register_buffer("sigma", torch.ones(out_features))

    # ------------------------------------------------------------------
    # Forward / normalize / denormalize
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return *normalized* value prediction."""
        return self.linear(x)

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        """Map normalized predictions back to the original scale."""
        return normalized * self.sigma + self.mu

    def normalize_targets(self, raw: torch.Tensor) -> torch.Tensor:
        """Normalize raw (un-scaled) targets for the value loss."""
        return (raw - self.mu) / self.sigma

    # ------------------------------------------------------------------
    # Core update + weight correction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_and_correct(self, raw_targets: torch.Tensor) -> None:
        """Update running statistics from *raw_targets* and correct weights.

        Called per critic mini-batch to track reward scale changes.
        """
        # ---------- save old stats ----------
        mu_old = self.mu.clone()
        sigma_old = self.sigma.clone()

        # ---------- EMA update ----------
        flat = raw_targets.reshape(-1, self.out_features)
        batch_mean = flat.mean(dim=0)
        batch_mean_sq = (flat ** 2).mean(dim=0)

        self._ema_mean.mul_(self.beta).add_(batch_mean, alpha=1.0 - self.beta)
        self._ema_mean_sq.mul_(self.beta).add_(batch_mean_sq, alpha=1.0 - self.beta)
        self._step.add_(1)

        # ---------- bias-corrected stats ----------
        bc = 1.0 - self.beta ** self._step.item()
        mean = self._ema_mean / bc
        mean_sq = self._ema_mean_sq / bc
        var = torch.clamp(mean_sq - mean ** 2, min=0.0)

        self.mu.copy_(mean)
        self.sigma.copy_(torch.sqrt(var + self.eps))

        # ---------- weight / bias correction ----------
        scale = sigma_old / self.sigma  # (out_features,)
        self.linear.weight.mul_(scale.unsqueeze(1))
        self.linear.bias.copy_(
            (sigma_old * self.linear.bias + mu_old - self.mu) / self.sigma
        )
