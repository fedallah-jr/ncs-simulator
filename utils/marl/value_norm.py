from __future__ import annotations

from typing import Optional, Sequence

import torch


class ValueNorm:
    def __init__(
        self,
        shape: Sequence[int] | int,
        eps: float = 1e-5,
        beta: float = 0.999,
        device: Optional[torch.device] = None,
    ) -> None:
        if isinstance(shape, int):
            if shape < 0:
                raise ValueError("shape must be non-negative")
            shape = () if shape == 0 else (shape,)
        self.shape = tuple(int(x) for x in shape)
        if any(dim <= 0 for dim in self.shape):
            raise ValueError("shape dimensions must be positive")
        self.eps = float(eps)
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must be in (0, 1)")
        self.device = device if device is not None else torch.device("cpu")
        self.beta = float(beta)

        self.mean = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        self.mean_sq = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        self.step = 0

    def update(self, values: torch.Tensor) -> None:
        if self.shape:
            if values.ndim < len(self.shape):
                raise ValueError("values must have at least as many dims as shape")
            if tuple(values.shape[-len(self.shape):]) != self.shape:
                raise ValueError("values trailing dimensions must match ValueNorm shape")
            flat = values.reshape(-1, *self.shape)
            batch_mean = flat.mean(dim=0)
            batch_mean_sq = (flat ** 2).mean(dim=0)
        else:
            flat = values.reshape(-1)
            batch_mean = flat.mean()
            batch_mean_sq = (flat ** 2).mean()

        self.mean = self.beta * self.mean + (1.0 - self.beta) * batch_mean
        self.mean_sq = self.beta * self.mean_sq + (1.0 - self.beta) * batch_mean_sq
        self.step += 1

    def _stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.step == 0:
            mean = self.mean
            var = torch.ones_like(self.mean)
            return mean, var
        bias_correction = 1.0 - self.beta ** self.step
        mean = self.mean / bias_correction
        mean_sq = self.mean_sq / bias_correction
        var = torch.clamp(mean_sq - mean ** 2, min=0.0)
        return mean, var

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        mean, var = self._stats()
        return (values - mean) / torch.sqrt(var + self.eps)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        mean, var = self._stats()
        return values * torch.sqrt(var + self.eps) + mean
