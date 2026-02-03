"""
Utilities for estimating reward normalization statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict, MutableMapping, Tuple
import uuid

import numpy as np


def _merge_running_stats(
    mean_a: float,
    m2_a: float,
    count_a: int,
    mean_b: float,
    m2_b: float,
    count_b: int,
) -> Tuple[float, float, int]:
    if count_a == 0:
        return mean_b, m2_b, count_b
    if count_b == 0:
        return mean_a, m2_a, count_a
    delta = mean_b - mean_a
    count = count_a + count_b
    mean = mean_a + delta * (count_b / count)
    m2 = m2_a + m2_b + (delta * delta) * (count_a * count_b / count)
    return mean, m2, count


def _new_namespace() -> str:
    return uuid.uuid4().hex


_RUNNING_NORMALIZER_NAMESPACE: str = _new_namespace()
_DEFAULT_SHARED_SYNC_INTERVAL = 32
_SHARED_RUNNING_NORMALIZER_SYNC_INTERVAL: int = _DEFAULT_SHARED_SYNC_INTERVAL


def set_running_normalizer_namespace(namespace: Optional[str] = None) -> str:
    global _RUNNING_NORMALIZER_NAMESPACE
    if namespace is None:
        _RUNNING_NORMALIZER_NAMESPACE = _new_namespace()
    else:
        _RUNNING_NORMALIZER_NAMESPACE = str(namespace)
    return _RUNNING_NORMALIZER_NAMESPACE


def _namespaced_key(key: str) -> str:
    return f"{_RUNNING_NORMALIZER_NAMESPACE}:{key}"


# VecNormalize-like running normalization (tracks return variance, scales reward).
@dataclass
class RunningRewardNormalizer:
    mean: float = 0.0
    m2: float = 0.0
    count: int = 0
    eps: float = 1e-8

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return max(float(np.sqrt(self.m2 / self.count)), self.eps)

    def __call__(self, reward: float) -> float:
        return reward / self.std


class SharedRunningRewardNormalizer(RunningRewardNormalizer):
    """
    Running reward normalizer backed by a shared, process-safe store.

    This mirrors RunningRewardNormalizer's interface but reads/writes its state
    from a shared mapping (e.g., multiprocessing.Manager().dict()) so that
    multiple processes can accumulate identical statistics.
    """

    def __init__(
        self,
        key: str,
        store: MutableMapping[str, Dict[str, float]],
        lock: Optional[Any] = None,
        eps: float = 1e-8,
        sync_interval: int = _DEFAULT_SHARED_SYNC_INTERVAL,
    ) -> None:
        super().__init__(mean=0.0, m2=0.0, count=0, eps=eps)
        self._key = key
        self._store = store
        self._lock = lock
        if sync_interval < 1:
            raise ValueError("sync_interval must be >= 1")
        self._sync_interval = int(sync_interval)
        self._local_updates = 0
        self._cached_mean = 0.0
        self._cached_m2 = 0.0
        self._cached_count = 0
        if self._lock is None:
            raise ValueError("SharedRunningRewardNormalizer requires a shared lock.")
        with self._lock:
            self._refresh_cached_state_locked()

    @property
    def std(self) -> float:
        mean, m2, count = self._combined_state()
        if count < 2:
            return 1.0
        return max(float(np.sqrt(m2 / count)), self.eps)

    def _ensure_state_locked(self) -> Dict[str, float]:
        state = self._store.get(self._key)
        if state is None:
            state = {"mean": 0.0, "m2": 0.0, "count": 0}
            self._store[self._key] = state
        return state

    def _read_state_locked(self) -> Tuple[float, float, int]:
        state = self._ensure_state_locked()
        mean = float(state.get("mean", 0.0))
        m2 = float(state.get("m2", 0.0))
        count = int(state.get("count", 0))
        return mean, m2, count

    def _write_state_locked(self, mean: float, m2: float, count: int) -> None:
        self._store[self._key] = {"mean": mean, "m2": m2, "count": count}

    def _refresh_cached_state_locked(self) -> None:
        mean, m2, count = self._read_state_locked()
        self._cached_mean = mean
        self._cached_m2 = m2
        self._cached_count = count

    def _combined_state(self) -> Tuple[float, float, int]:
        return _merge_running_stats(
            self._cached_mean,
            self._cached_m2,
            self._cached_count,
            self.mean,
            self.m2,
            self.count,
        )

    def _reset_local_state(self) -> None:
        self.mean = 0.0
        self.m2 = 0.0
        self.count = 0
        self._local_updates = 0

    def _flush_local_updates(self) -> None:
        if self.count == 0:
            self._local_updates = 0
            return
        with self._lock:
            shared_mean, shared_m2, shared_count = self._read_state_locked()
            mean, m2, count = _merge_running_stats(
                shared_mean,
                shared_m2,
                shared_count,
                self.mean,
                self.m2,
                self.count,
            )
            self._write_state_locked(mean, m2, count)
            self._cached_mean = mean
            self._cached_m2 = m2
            self._cached_count = count
        self._reset_local_state()

    def update(self, value: float) -> None:
        super().update(value)
        self._local_updates += 1
        if self._local_updates >= self._sync_interval:
            self._flush_local_updates()


_SHARED_RUNNING_NORMALIZERS: Dict[str, RunningRewardNormalizer] = {}
_SHARED_RUNNING_NORMALIZER_STORE: Optional[MutableMapping[str, Dict[str, float]]] = None
_SHARED_RUNNING_NORMALIZER_LOCK: Optional[Any] = None


def get_shared_running_normalizer(key: str) -> RunningRewardNormalizer:
    namespaced_key = _namespaced_key(key)
    normalizer = _SHARED_RUNNING_NORMALIZERS.get(namespaced_key)
    if normalizer is not None:
        return normalizer

    if _SHARED_RUNNING_NORMALIZER_STORE is not None:
        normalizer = SharedRunningRewardNormalizer(
            key=namespaced_key,
            store=_SHARED_RUNNING_NORMALIZER_STORE,
            lock=_SHARED_RUNNING_NORMALIZER_LOCK,
            sync_interval=_SHARED_RUNNING_NORMALIZER_SYNC_INTERVAL,
        )
    else:
        normalizer = RunningRewardNormalizer()

    _SHARED_RUNNING_NORMALIZERS[namespaced_key] = normalizer
    return normalizer


def configure_shared_running_normalizers(
    store: Optional[MutableMapping[str, Dict[str, float]]],
    lock: Optional[Any] = None,
    *,
    sync_interval: int = _DEFAULT_SHARED_SYNC_INTERVAL,
    namespace: Optional[str] = None,
    reset_store: bool = True,
) -> None:
    """
    Configure the backend store used for shared running reward normalizers.

    Pass a multiprocessing.Manager().dict() and a shared lock when using
    multiple processes so statistics stay synchronized across workers. Passing
    None reverts to a process-local dictionary. The sync interval batches
    updates to reduce IPC overhead, and the namespace isolates runs. Set
    reset_store=False when attaching workers to an existing shared store to
    avoid clearing accumulated statistics.
    """
    global _SHARED_RUNNING_NORMALIZER_STORE, _SHARED_RUNNING_NORMALIZER_LOCK
    global _SHARED_RUNNING_NORMALIZER_SYNC_INTERVAL
    if store is not None and lock is None:
        raise ValueError("Shared normalizers require a shared lock.")
    if sync_interval < 1:
        raise ValueError("sync_interval must be >= 1")
    _SHARED_RUNNING_NORMALIZER_STORE = store
    _SHARED_RUNNING_NORMALIZER_LOCK = lock
    _SHARED_RUNNING_NORMALIZER_SYNC_INTERVAL = int(sync_interval)
    if store is None:
        set_running_normalizer_namespace(namespace)
    else:
        set_running_normalizer_namespace(namespace or "shared")
    _SHARED_RUNNING_NORMALIZERS.clear()
    if reset_store and _SHARED_RUNNING_NORMALIZER_STORE is not None:
        if _SHARED_RUNNING_NORMALIZER_LOCK is not None:
            with _SHARED_RUNNING_NORMALIZER_LOCK:
                _SHARED_RUNNING_NORMALIZER_STORE.clear()
        else:
            _SHARED_RUNNING_NORMALIZER_STORE.clear()


def reset_shared_running_normalizers() -> None:
    _SHARED_RUNNING_NORMALIZERS.clear()
    if _SHARED_RUNNING_NORMALIZER_STORE is None:
        set_running_normalizer_namespace(None)
    if _SHARED_RUNNING_NORMALIZER_STORE is not None:
        if _SHARED_RUNNING_NORMALIZER_LOCK is not None:
            with _SHARED_RUNNING_NORMALIZER_LOCK:
                _SHARED_RUNNING_NORMALIZER_STORE.clear()
        else:
            _SHARED_RUNNING_NORMALIZER_STORE.clear()
