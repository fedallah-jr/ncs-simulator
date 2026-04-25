"""
Orchestrate the finalized experiment matrix in Python instead of bash.

Replaces run_experiment_1..8 with a single numbered registry of 20 experiments
that can be split across machines for parallel execution. Each ID is fully
self-contained: train -> policy_tester -> rename -> zip.

Usage
-----

    # List all experiments and exit.
    python -m tools.run_experiments --list

    # Run a specific subset (CSV or ranges).
    python -m tools.run_experiments --ids 1,2,3
    python -m tools.run_experiments --ids 7-12
    python -m tools.run_experiments --ids 1,4-6,13

    # Show the constructed commands without executing them.
    python -m tools.run_experiments --ids 1-3 --dry-run

    # Override defaults.
    python -m tools.run_experiments --ids 1-6 --seed 0 \
        --output-root outputs --num-policy-test-seeds 250

    # Skip the policy_tester and zip steps (e.g. when iterating).
    python -m tools.run_experiments --ids 1 --skip-policy-test --skip-zip

Design notes
------------

* All experiments use the heterogeneous config (configs/marl_absolute_plants_hetero.json)
  and seed 0 (override via --seed).
* Cat 1 (IDs 1-6): IQL/QMIX/VDN/MAPPO/HAPPO/HASAC at 15M, mirroring run_experiment_{1..6}.
  Q-learners get --double-q; --n-step 3 is the default for IQL/QMIX/VDN and is
  set explicitly for HASAC (HASAC default is 20). MAPPO/HAPPO use GAE so n-step
  does not apply; they rely on ValueNorm via the default code path.
* Cat 2 (IDs 7-12): NDQ comm-embed-dim sweep {5,10,15} x {vdn,qmix} at 30M,
  mirroring run_experiment_8. NDQ does not expose --feature-norm/--layer-norm/--n-step.
* Cat 3 (IDs 13-16): VDN/QMIX/HAPPO/HASAC + 8-bit hand-crafted error comm at 15M.
* Cat 4 (IDs 17-20): VDN/QMIX/HAPPO/HASAC + 4-bit hand-crafted error comm + 4-bit age comm at 15M.
* The VoU (Value of Update) quantile search runs once and produces edges for
  bits_1 through bits_8 in a single output. Cached under
  outputs/_shared/vou_search/ and reused across all comm experiments.
* Per-experiment layout:
    outputs/experiments_<ID>/
      <NAME>/                   # renamed from {algo}_0
        best_model.pt, latest_model.pt, config.json, ...
        policy_tests/           # policy_tester output
      logs/
        train.log
        policy_test.log
* Each experiment is zipped to outputs/experiments_<ID>.zip.
* Failures in one experiment do not abort the rest; a summary is printed at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = "configs/marl_absolute_plants_hetero.json"
DEFAULT_OUTPUT_ROOT = "outputs"
SHARED_DIR_NAME = "_shared"
VOU_CACHE_DIRNAME = "vou_search"

CAT1_TIMESTEPS = 15_000_000
CAT2_TIMESTEPS = 30_000_000
CAT3_TIMESTEPS = 15_000_000
CAT4_TIMESTEPS = 15_000_000

# Q-learner bash defaults (run_experiment_1..3).
QLEARNER_BASE_ARGS: List[str] = [
    "--double-q",
    "--batch-size", "512",
    "--learning-rate", "5e-4",
    "--no-normalize-obs",
    "--feature-norm",
    "--layer-norm",
    "--eval-freq", "40000",
    "--n-eval-episodes", "80",
    "--n-eval-envs", "8",
    "--n-envs", "8",
    "--episode-length", "250",
]

# MAPPO/HAPPO bash defaults (run_experiment_4/5).
ON_POLICY_BASE_ARGS: List[str] = [
    "--no-normalize-obs",
    "--feature-norm",
    "--layer-norm",
    "--eval-freq", "40000",
    "--n-eval-episodes", "80",
    "--n-envs", "8",
    "--episode-length", "250",
]

# HASAC bash defaults (run_experiment_6).
HASAC_BASE_ARGS: List[str] = [
    "--no-normalize-obs",
    "--feature-norm",
    "--layer-norm",
    "--value-norm",
    "--n-step", "3",
    "--eval-freq", "40000",
    "--n-eval-episodes", "80",
    "--n-envs", "20",
    "--episode-length", "250",
]

# NDQ bash defaults (run_experiment_8). No --feature-norm/--layer-norm/--n-step exposed.
NDQ_BASE_ARGS: List[str] = [
    "--double-q",
    "--no-normalize-obs",
    "--eval-freq", "40000",
    "--n-eval-episodes", "80",
    "--n-envs", "8",
    "--episode-length", "250",
]

# Hand-crafted communication is built on top of the per-algo base args,
# matching the bash --error_comm / --age_comm pattern.
HANDCRAFT_BITS_8 = 8
HANDCRAFT_BITS_4 = 4
AGE_BITS_4 = 4

# VoU search defaults (mirror run_experiment_1..6 defaults).
VOU_SEARCH_DEFAULTS = {
    "iters": 12,
    "samples_per_iter": 8,
    "eval_num_seeds": 32,
    "threshold_min": 0.01,
    "threshold_max": 10.0,
    "collect_bits": 8,  # produces edges for bits_1..bits_8 in a single search
}


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------


@dataclass
class Experiment:
    """One end-to-end experiment: train -> policy_test -> zip."""

    id: int
    name: str                # human-readable folder name (e.g. "IQL_doubleq_15mil")
    module: str              # python module to invoke (e.g. "algorithms.marl_iql")
    algo_label: str          # subdir prefix used by prepare_run_directory (e.g. "iql")
    total_timesteps: int     # training steps
    extra_args: List[str] = field(default_factory=list)
    needs_vou_edges: bool = False
    error_comm_bits: int = 0  # 0 = disabled
    age_comm_bits: int = 0    # 0 = disabled


def _eps_decay(total: int) -> str:
    """60% of total timesteps, matching the bash scripts."""
    return str(total * 60 // 100)


def _qlearner_args(total: int) -> List[str]:
    return list(QLEARNER_BASE_ARGS) + [
        "--total-timesteps", str(total),
        "--epsilon-decay-steps", _eps_decay(total),
    ]


def _ndq_args(total: int, mixer: str, comm_dim: int) -> List[str]:
    return list(NDQ_BASE_ARGS) + [
        "--total-timesteps", str(total),
        "--epsilon-decay-steps", _eps_decay(total),
        "--mixer", mixer,
        "--comm-embed-dim", str(comm_dim),
    ]


def _on_policy_args(total: int) -> List[str]:
    return list(ON_POLICY_BASE_ARGS) + ["--total-timesteps", str(total)]


def _hasac_args(total: int) -> List[str]:
    return list(HASAC_BASE_ARGS) + ["--total-timesteps", str(total)]


def _comm_overrides(error_bits: int, age_bits: int) -> List[str]:
    """Build the --error_comm / --age_comm flags. error_comm_edges injected later."""
    args: List[str] = []
    if error_bits > 0:
        args += ["--error_comm", "--set", f"observation.error_comm_bits={error_bits}"]
    if age_bits > 0:
        args += ["--age_comm", "--set", f"observation.age_comm_bits={age_bits}"]
    return args


def build_registry() -> List[Experiment]:
    exps: List[Experiment] = []

    # ----- Cat 1: 6 algos at 15M -----
    exps.append(Experiment(
        id=1, name="IQL_doubleq_15mil",
        module="algorithms.marl_iql", algo_label="iql",
        total_timesteps=CAT1_TIMESTEPS,
        extra_args=_qlearner_args(CAT1_TIMESTEPS),
    ))
    exps.append(Experiment(
        id=2, name="QMIX_doubleq_15mil",
        module="algorithms.marl_qmix", algo_label="qmix",
        total_timesteps=CAT1_TIMESTEPS,
        extra_args=_qlearner_args(CAT1_TIMESTEPS),
    ))
    exps.append(Experiment(
        id=3, name="VDN_doubleq_15mil",
        module="algorithms.marl_vdn", algo_label="vdn",
        total_timesteps=CAT1_TIMESTEPS,
        extra_args=_qlearner_args(CAT1_TIMESTEPS),
    ))
    exps.append(Experiment(
        id=4, name="MAPPO_15mil",
        module="algorithms.marl_mappo", algo_label="mappo",
        total_timesteps=CAT1_TIMESTEPS,
        extra_args=_on_policy_args(CAT1_TIMESTEPS),
    ))
    exps.append(Experiment(
        id=5, name="HAPPO_15mil",
        module="algorithms.marl_happo", algo_label="happo",
        total_timesteps=CAT1_TIMESTEPS,
        extra_args=_on_policy_args(CAT1_TIMESTEPS),
    ))
    exps.append(Experiment(
        id=6, name="HASAC_15mil",
        module="algorithms.marl_hasac", algo_label="hasac",
        total_timesteps=CAT1_TIMESTEPS,
        extra_args=_hasac_args(CAT1_TIMESTEPS),
    ))

    # ----- Cat 2: NDQ comm sweep at 30M -----
    next_id = 7
    for comm_dim in (5, 10, 15):
        for mixer in ("vdn", "qmix"):
            exps.append(Experiment(
                id=next_id,
                name=f"NDQ_{comm_dim}dim_{mixer}_30mil",
                module="algorithms.marl_ndq",
                algo_label="marl_ndq",
                total_timesteps=CAT2_TIMESTEPS,
                extra_args=_ndq_args(CAT2_TIMESTEPS, mixer, comm_dim),
            ))
            next_id += 1

    # ----- Cat 3: VDN/QMIX/HAPPO/HASAC + 8-bit hand-crafted error comm -----
    cat34_specs = [
        ("VDN", "algorithms.marl_vdn", "vdn", _qlearner_args),
        ("QMIX", "algorithms.marl_qmix", "qmix", _qlearner_args),
        ("HAPPO", "algorithms.marl_happo", "happo", _on_policy_args),
        ("HASAC", "algorithms.marl_hasac", "hasac", _hasac_args),
    ]

    for prefix, module, label, base_fn in cat34_specs:
        exps.append(Experiment(
            id=next_id,
            name=f"{prefix}_8bithand_15mil",
            module=module, algo_label=label,
            total_timesteps=CAT3_TIMESTEPS,
            extra_args=base_fn(CAT3_TIMESTEPS) + _comm_overrides(HANDCRAFT_BITS_8, 0),
            needs_vou_edges=True,
            error_comm_bits=HANDCRAFT_BITS_8,
        ))
        next_id += 1

    # ----- Cat 4: VDN/QMIX/HAPPO/HASAC + 4-bit hand-crafted + 4-bit age comm -----
    for prefix, module, label, base_fn in cat34_specs:
        exps.append(Experiment(
            id=next_id,
            name=f"{prefix}_4bithand_4bitage_15mil",
            module=module, algo_label=label,
            total_timesteps=CAT4_TIMESTEPS,
            extra_args=(
                base_fn(CAT4_TIMESTEPS)
                + _comm_overrides(HANDCRAFT_BITS_4, AGE_BITS_4)
            ),
            needs_vou_edges=True,
            error_comm_bits=HANDCRAFT_BITS_4,
            age_comm_bits=AGE_BITS_4,
        ))
        next_id += 1

    return exps


REGISTRY = build_registry()


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def parse_id_spec(spec: str) -> List[int]:
    """Parse '1,4-6,9' into [1,4,5,6,9]."""
    ids: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            for i in range(int(lo), int(hi) + 1):
                ids.append(i)
        else:
            ids.append(int(chunk))
    seen = set()
    unique: List[int] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    return unique


def print_registry() -> None:
    header = f"{'ID':<4}{'NAME':<38}{'MODULE':<28}{'STEPS':<14}{'COMM'}"
    print(header)
    print("-" * len(header))
    for exp in REGISTRY:
        comm_bits = []
        if exp.error_comm_bits > 0:
            comm_bits.append(f"err{exp.error_comm_bits}")
        if exp.age_comm_bits > 0:
            comm_bits.append(f"age{exp.age_comm_bits}")
        comm = "+".join(comm_bits) if comm_bits else "-"
        print(f"{exp.id:<4}{exp.name:<38}{exp.module:<28}{exp.total_timesteps:<14,}{comm}")


# ---------------------------------------------------------------------------
# Subprocess execution with tee-style log capture
# ---------------------------------------------------------------------------


def run_command(cmd: Sequence[str], log_path: Path, *, cwd: Optional[Path] = None,
                dry_run: bool = False) -> int:
    """Run a subprocess, mirroring stdout/stderr to both the terminal and a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pretty = " ".join(cmd)
    print(f"\n$ {pretty}\n  -> log: {log_path}")
    if dry_run:
        return 0

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"$ {pretty}\n\n")
        log_f.flush()
        proc = subprocess.Popen(
            list(cmd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(cwd) if cwd else None, env=env, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
            log_f.flush()
        proc.wait()
        return proc.returncode


# ---------------------------------------------------------------------------
# VoU search (one shared search produces edges for bits_1..bits_8)
# ---------------------------------------------------------------------------


def ensure_vou_edges(
    config: str, output_root: Path, *,
    force: bool = False, dry_run: bool = False,
) -> Optional[Path]:
    """
    Run tools.search_vou_threshold once and cache the result. Returns the path
    to quantile_edges.json. Reused across all comm experiments.
    """
    cache_dir = output_root / SHARED_DIR_NAME / VOU_CACHE_DIRNAME
    edges_path = cache_dir / "quantile_edges.json"
    summary_path = cache_dir / "search_summary.json"

    if not force and edges_path.exists() and summary_path.exists():
        print(f"[vou] reusing cached search at {cache_dir}")
        return edges_path

    if force and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "tools.search_vou_threshold",
        "--config", config,
        "--threshold-min", str(VOU_SEARCH_DEFAULTS["threshold_min"]),
        "--threshold-max", str(VOU_SEARCH_DEFAULTS["threshold_max"]),
        "--iters", str(VOU_SEARCH_DEFAULTS["iters"]),
        "--samples-per-iter", str(VOU_SEARCH_DEFAULTS["samples_per_iter"]),
        "--eval-num-seeds", str(VOU_SEARCH_DEFAULTS["eval_num_seeds"]),
        "--collect-bits", str(VOU_SEARCH_DEFAULTS["collect_bits"]),
        "--output-dir", str(cache_dir),
    ]
    log_path = cache_dir / "search.log"
    rc = run_command(cmd, log_path, cwd=PROJECT_ROOT, dry_run=dry_run)
    if rc != 0:
        raise RuntimeError(f"VoU search failed with exit code {rc}")
    if not dry_run and not edges_path.exists():
        raise RuntimeError(f"VoU search did not produce {edges_path}")
    return edges_path


def load_edges_for_bits(edges_path: Path, bits: int) -> List[float]:
    with edges_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    bucket = data.get("edges", {}).get(f"bits_{bits}")
    if bucket is None:
        raise KeyError(f"quantile_edges.json missing bits_{bits} entry")
    edges = bucket.get("edges")
    if edges is None:
        raise KeyError(f"quantile_edges.json bits_{bits} missing 'edges' key")
    return list(edges)


# ---------------------------------------------------------------------------
# Per-experiment pipeline
# ---------------------------------------------------------------------------


def find_existing_run_dir(parent: Path, label: str) -> Optional[Path]:
    """Find the most recently created {label}_N subdir under parent (post-train)."""
    if not parent.exists():
        return None
    candidates = sorted(parent.glob(f"{label}_*"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def find_next_run_dir(parent: Path, label: str) -> Path:
    """Predict the next sequential run dir name ({label}_N) before training."""
    parent.mkdir(parents=True, exist_ok=True)
    idx = 0
    while True:
        candidate = parent / f"{label}_{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def build_train_command(
    exp: Experiment, *, config: str, run_root: Path, seed: int,
    edges: Optional[List[float]],
) -> List[str]:
    cmd = [
        sys.executable, "-m", exp.module,
        "--config", config,
        "--output-root", str(run_root),
        "--seed", str(seed),
    ]
    cmd.extend(exp.extra_args)
    if exp.needs_vou_edges and edges is not None:
        cmd.extend(["--set", f"observation.error_comm_edges={json.dumps(edges)}"])
    return cmd


def build_policy_test_command(
    config: str, run_dir: Path, num_seeds: int,
) -> List[str]:
    return [
        sys.executable, "-m", "tools.policy_tester",
        "--config", config,
        "--policy", str(run_dir / "best_model.pt"),
        "--policy-type", "marl_torch",
        "--num-seeds", str(num_seeds),
        "--output-dir", str(run_dir / "policy_tests"),
    ]


def zip_directory(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    base = src_dir.parent
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            zf.write(p, p.relative_to(base))


def run_experiment(
    exp: Experiment, *, config: str, output_root: Path, seed: int,
    num_policy_test_seeds: int, edges_path: Optional[Path],
    skip_policy_test: bool, skip_zip: bool, dry_run: bool,
) -> None:
    """Train -> rename -> policy_test -> zip for a single experiment."""
    run_root = output_root / f"experiments_{exp.id}"
    if run_root.exists():
        raise RuntimeError(
            f"Output dir already exists: {run_root}. "
            f"Move it aside or delete it before re-running."
        )
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Resolve edges payload for this experiment, if needed.
    edges: Optional[List[float]] = None
    if exp.needs_vou_edges:
        if edges_path is None:
            raise RuntimeError(f"Experiment {exp.id} needs VoU edges but none available")
        if dry_run and not edges_path.exists():
            # VoU search hasn't actually run; inject a placeholder so the printed
            # command is still representative.
            edges = [f"<bits_{exp.error_comm_bits}_edges_from_vou>"]  # type: ignore[list-item]
        else:
            edges = load_edges_for_bits(edges_path, exp.error_comm_bits)

    # Train.
    expected_train_dir = find_next_run_dir(run_root, exp.algo_label)
    train_cmd = build_train_command(
        exp, config=config, run_root=run_root, seed=seed, edges=edges,
    )
    rc = run_command(train_cmd, logs_dir / "train.log", cwd=PROJECT_ROOT, dry_run=dry_run)
    if rc != 0:
        raise RuntimeError(f"Training exited with code {rc}")

    # Rename {algo_label}_N -> exp.name.
    actual_dir: Path
    if dry_run:
        actual_dir = expected_train_dir
    else:
        if expected_train_dir.exists():
            actual_dir = expected_train_dir
        else:
            located = find_existing_run_dir(run_root, exp.algo_label)
            if located is None:
                raise RuntimeError(
                    f"Could not locate trained run dir under {run_root} "
                    f"(expected prefix: {exp.algo_label}_*)"
                )
            actual_dir = located
        final_dir = run_root / exp.name
        if final_dir.exists():
            raise RuntimeError(f"Target run dir already exists: {final_dir}")
        actual_dir.rename(final_dir)
        actual_dir = final_dir
    print(f"[{exp.name}] training dir: {actual_dir}")

    # Policy testing.
    if skip_policy_test:
        print(f"[{exp.name}] skipping policy_tester (per --skip-policy-test)")
    else:
        pt_cmd = build_policy_test_command(config, actual_dir, num_policy_test_seeds)
        rc = run_command(pt_cmd, logs_dir / "policy_test.log", cwd=PROJECT_ROOT, dry_run=dry_run)
        if rc != 0:
            raise RuntimeError(f"policy_tester exited with code {rc}")

    # Zip.
    if skip_zip:
        print(f"[{exp.name}] skipping zip (per --skip-zip)")
    else:
        zip_path = output_root / f"experiments_{exp.id}.zip"
        if dry_run:
            print(f"[{exp.name}] would zip {run_root} -> {zip_path}")
        else:
            zip_directory(run_root, zip_path)
            print(f"[{exp.name}] zipped -> {zip_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Numbered experiment orchestrator (replaces run_experiment_*.sh).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--list", action="store_true", help="Print the experiment registry and exit.")
    p.add_argument("--ids", default=None,
                   help="Experiment IDs to run, e.g. '1,2,3' or '7-12' or '1,4-6,9'.")
    p.add_argument("--config", default=DEFAULT_CONFIG, help=f"Config path (default: {DEFAULT_CONFIG}).")
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                   help=f"Output root directory (default: {DEFAULT_OUTPUT_ROOT}).")
    p.add_argument("--seed", type=int, default=0, help="Training seed (default: 0).")
    p.add_argument("--num-policy-test-seeds", type=int, default=250,
                   help="Seeds passed to policy_tester (default: 250, the tool default).")
    p.add_argument("--skip-policy-test", action="store_true",
                   help="Skip the post-training policy_tester step.")
    p.add_argument("--skip-zip", action="store_true",
                   help="Skip the post-training zip step.")
    p.add_argument("--skip-vou", action="store_true",
                   help="Skip the VoU search step (use cached edges only; fail if missing).")
    p.add_argument("--force-vou", action="store_true",
                   help="Re-run VoU search even if a cached result exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the constructed commands without executing them.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.list:
        print_registry()
        return 0

    if not args.ids:
        print("error: --ids is required (use --list to see available IDs)", file=sys.stderr)
        return 2

    requested = parse_id_spec(args.ids)
    by_id = {exp.id: exp for exp in REGISTRY}
    missing = [i for i in requested if i not in by_id]
    if missing:
        print(f"error: unknown experiment IDs: {missing}", file=sys.stderr)
        return 2

    output_root = Path(args.output_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.exists():
        print(f"error: config not found: {config_path}", file=sys.stderr)
        return 2

    experiments = [by_id[i] for i in requested]
    needs_edges = any(e.needs_vou_edges for e in experiments)

    edges_path: Optional[Path] = None
    if needs_edges:
        if args.skip_vou:
            cache = output_root / SHARED_DIR_NAME / VOU_CACHE_DIRNAME / "quantile_edges.json"
            if not cache.exists():
                print(f"error: --skip-vou set but cached edges missing: {cache}", file=sys.stderr)
                return 2
            edges_path = cache
            print(f"[vou] using cached edges at {edges_path}")
        else:
            edges_path = ensure_vou_edges(
                str(config_path), output_root,
                force=args.force_vou, dry_run=args.dry_run,
            )

    summary: List[tuple] = []  # (id, name, status, elapsed_sec, error)
    for exp in experiments:
        print("\n" + "=" * 70)
        print(f"== Experiment {exp.id}: {exp.name}")
        print("=" * 70)
        t0 = time.time()
        try:
            run_experiment(
                exp, config=str(config_path), output_root=output_root,
                seed=args.seed,
                num_policy_test_seeds=args.num_policy_test_seeds,
                edges_path=edges_path,
                skip_policy_test=args.skip_policy_test,
                skip_zip=args.skip_zip, dry_run=args.dry_run,
            )
            summary.append((exp.id, exp.name, "OK", time.time() - t0, ""))
        except Exception as exc:  # noqa: BLE001 - surface to summary rather than crash
            err = f"{type(exc).__name__}: {exc}"
            print(f"\n[ERROR] {exp.name}: {err}", file=sys.stderr)
            summary.append((exp.id, exp.name, "FAIL", time.time() - t0, err))

    print("\n" + "=" * 70)
    print("== Summary")
    print("=" * 70)
    print(f"{'ID':<4}{'NAME':<38}{'STATUS':<8}{'TIME':<12}{'ERROR'}")
    for sid, sname, sstatus, selapsed, serr in summary:
        print(f"{sid:<4}{sname:<38}{sstatus:<8}{selapsed:<12.1f}{serr}")

    failures = sum(1 for s in summary if s[2] != "OK")
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
