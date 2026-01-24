from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

import numpy as np

from utils.bc.dataset import BCDataset


@dataclass
class BCPretrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    use_agent_id: bool
    n_agents: int


@dataclass
class BCPretrainResult:
    params: Any
    metrics: Dict[str, float]


class ActorBCAdapter(Protocol):
    def pretrain(
        self,
        params: Any,
        dataset: BCDataset,
        config: BCPretrainConfig,
        rng: np.random.Generator,
    ) -> BCPretrainResult:
        ...


def pretrain_actor(
    adapter: ActorBCAdapter,
    params: Any,
    dataset: BCDataset,
    config: BCPretrainConfig,
    rng: np.random.Generator,
) -> BCPretrainResult:
    return adapter.pretrain(params, dataset, config, rng)


class JaxActorBCAdapter:
    def __init__(
        self,
        apply_fn: Any,
        n_agents: int,
        use_agent_id: bool,
    ) -> None:
        self.apply_fn = apply_fn
        self.n_agents = int(n_agents)
        self.use_agent_id = bool(use_agent_id)
        self._agent_eye = np.eye(self.n_agents, dtype=np.float32) if self.use_agent_id else None

    def _augment_obs(self, obs: np.ndarray, agent_idx: np.ndarray) -> np.ndarray:
        if not self.use_agent_id:
            return obs
        if self._agent_eye is None:
            raise RuntimeError("Agent id eye not initialized.")
        return np.concatenate([obs, self._agent_eye[agent_idx]], axis=1)

    def pretrain(
        self,
        params: Any,
        dataset: BCDataset,
        config: BCPretrainConfig,
        rng: np.random.Generator,
    ) -> BCPretrainResult:
        if config.n_agents != self.n_agents:
            raise ValueError("config.n_agents does not match adapter n_agents")
        if config.use_agent_id != self.use_agent_id:
            raise ValueError("config.use_agent_id does not match adapter use_agent_id")
        if config.epochs <= 0:
            raise ValueError("epochs must be positive")
        if config.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if config.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

        import jax
        import jax.numpy as jnp
        import optax

        optimizer = optax.adam(config.learning_rate)
        opt_state = optimizer.init(params)

        def loss_fn(params_local: Any, obs: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
            logits = self.apply_fn(params_local, obs)
            return optax.softmax_cross_entropy_with_integer_labels(logits, actions).mean()

        @jax.jit
        def train_step(
            params_local: Any,
            opt_state_local: Any,
            obs: jnp.ndarray,
            actions: jnp.ndarray,
        ) -> tuple[Any, Any, jnp.ndarray]:
            loss, grads = jax.value_and_grad(loss_fn)(params_local, obs, actions)
            updates, opt_state_next = optimizer.update(grads, opt_state_local, params_local)
            params_next = optax.apply_updates(params_local, updates)
            return params_next, opt_state_next, loss

        total_loss = 0.0
        total_batches = 0
        for _epoch in range(config.epochs):
            for obs_mb, actions_mb, agent_idx in dataset.iter_actor_batches(config.batch_size, rng):
                obs_mb = self._augment_obs(obs_mb, agent_idx)
                params, opt_state, loss = train_step(
                    params,
                    opt_state,
                    jnp.asarray(obs_mb),
                    jnp.asarray(actions_mb),
                )
                total_loss += float(loss)
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        metrics = {
            "loss": float(avg_loss),
            "batches": float(total_batches),
        }
        return BCPretrainResult(params=params, metrics=metrics)
