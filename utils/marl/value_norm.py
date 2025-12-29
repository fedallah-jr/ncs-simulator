from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch


class ValueNorm:
    def __init__(
        self,
        shape: Sequence[int] | int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
    ) -> None:
        if isinstance(shape, int):
            shape = (shape,)
        if not shape:
            raise ValueError("shape must be non-empty")
        self.shape: Tuple[int, ...] = tuple(int(x) for x in shape)
        self.eps = float(eps)
        self.device = device if device is not None else torch.device("cpu")

        self.mean = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        self.var = torch.ones(self.shape, dtype=torch.float32, device=self.device)
        self.count = torch.tensor(self.eps, dtype=torch.float32, device=self.device)

    def update(self, values: torch.Tensor) -> None:
        if values.ndim < len(self.shape):
            raise ValueError("values must have at least as many dims as shape")
        if tuple(values.shape[-len(self.shape):]) != self.shape:
            raise ValueError("values trailing dimensions must match ValueNorm shape")

        flat = values.reshape(-1, *self.shape)
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(flat.shape[0]), device=self.device)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / torch.sqrt(self.var + self.eps)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * torch.sqrt(self.var + self.eps) + self.mean
