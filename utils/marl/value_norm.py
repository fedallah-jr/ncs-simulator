from __future__ import annotations

from typing import Optional, Sequence

import torch


class ValueNorm:
    def __init__(
        self,
        shape: Sequence[int] | int,
        eps: float = 1e-5,
        beta: float = 0.99999,
        per_element_update: bool = False,
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
        self.per_element_update = bool(per_element_update)

        self.mean = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        self.mean_sq = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        self.debiasing_term = torch.tensor(0.0, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        values = values.to(device=self.device, dtype=torch.float32)
        if self.shape:
            if values.ndim < len(self.shape):
                raise ValueError("values must have at least as many dims as shape")
            if tuple(values.shape[-len(self.shape):]) != self.shape:
                raise ValueError("values trailing dimensions must match ValueNorm shape")
            flat = values.reshape(-1, *self.shape)
            batch_mean = flat.mean(dim=0)
            batch_mean_sq = (flat ** 2).mean(dim=0)
            batch_size = int(flat.shape[0])
        else:
            flat = values.reshape(-1)
            batch_mean = flat.mean()
            batch_mean_sq = (flat ** 2).mean()
            batch_size = int(flat.shape[0])

        if self.per_element_update:
            weight = self.beta ** batch_size
        else:
            weight = self.beta

        self.mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.mean_sq.mul_(weight).add_(batch_mean_sq * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def _stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        debiased_mean = self.mean / torch.clamp(self.debiasing_term, min=self.eps)
        debiased_mean_sq = self.mean_sq / torch.clamp(self.debiasing_term, min=self.eps)
        var = torch.clamp(debiased_mean_sq - debiased_mean ** 2, min=1e-2)
        return debiased_mean, var

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        values = values.to(device=self.device, dtype=torch.float32)
        mean, var = self._stats()
        return (values - mean) / torch.sqrt(var)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        values = values.to(device=self.device, dtype=torch.float32)
        mean, var = self._stats()
        return values * torch.sqrt(var) + mean

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.detach().cpu().clone(),
            "mean_sq": self.mean_sq.detach().cpu().clone(),
            "debiasing_term": self.debiasing_term.detach().cpu().clone(),
            "beta": self.beta,
            "per_element_update": self.per_element_update,
        }

    def load_state_dict(self, d: dict) -> None:
        if "beta" in d:
            self.beta = float(d["beta"])
        elif "step" in d:
            # Legacy checkpoints did not persist beta and defaulted to 0.999.
            self.beta = 0.999
        if "per_element_update" in d:
            self.per_element_update = bool(d["per_element_update"])

        self.mean = d["mean"].to(device=self.device, dtype=torch.float32)
        self.mean_sq = d["mean_sq"].to(device=self.device, dtype=torch.float32)
        if "debiasing_term" in d:
            self.debiasing_term = d["debiasing_term"].to(
                device=self.device, dtype=torch.float32
            )
        elif "step" in d:
            step = int(d["step"])
            debias = 1.0 - self.beta ** step
            self.debiasing_term = torch.tensor(
                debias, dtype=torch.float32, device=self.device
            )
        else:
            raise KeyError("ValueNorm state_dict must include debiasing_term or step")
