"""
Bundle a run_experiments / policy_tester batch directory into a results-only zip.

Skips PyTorch checkpoints (*.pt: best_model, latest_model, best_train_model,
training_state) and keeps everything else: leaderboard CSVs, per-model
policy_tests/ aggregates, training_rewards.csv, evaluation_drop_stats.csv,
config.json, perfect_comm_baseline/, and logs/.

Usage
-----

    python tools/extract_results.py --models-root outputs/experiments_1-6

    # custom output path:
    python tools/extract_results.py \
        --models-root outputs/experiments_1-6 \
        --output /tmp/cat1_results.zip
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bundle policy_tester / run_experiments results without *.pt checkpoints."
    )
    ap.add_argument(
        "--models-root", required=True, type=Path,
        help="Path to the evaluation/batch folder, e.g. outputs/experiments_1-6.",
    )
    ap.add_argument(
        "--output", type=Path, default=None,
        help="Output zip path (default: <models-root>_results.zip next to it).",
    )
    args = ap.parse_args()

    src = args.models_root.resolve()
    if not src.is_dir():
        print(f"error: not a directory: {src}", file=sys.stderr)
        return 2

    out = args.output.resolve() if args.output else src.parent / f"{src.name}_results.zip"
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = skipped = 0
    kept_bytes = skipped_bytes = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src.rglob("*")):
            if not p.is_file():
                continue
            sz = p.stat().st_size
            if p.suffix == ".pt":
                skipped += 1
                skipped_bytes += sz
                continue
            zf.write(p, p.relative_to(src.parent))
            kept += 1
            kept_bytes += sz

    print(f"wrote {out}")
    print(f"  kept    {kept:>5} files  ({_fmt_bytes(kept_bytes)})")
    print(f"  skipped {skipped:>5} *.pt  ({_fmt_bytes(skipped_bytes)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
