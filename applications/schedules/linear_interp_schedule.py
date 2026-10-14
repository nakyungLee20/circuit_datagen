"""Linear-ratio helpers and diagnostics for the Exp3 ordered trainer.

Training itself is implemented in :mod:`ordered_schedule_train`, where the
target ratios are evaluated at every physical training batch and the resulting
ordered examples are consumed with ``shuffle=False``.  This module intentionally
does not emit a pool and does not claim that three midpoint stages are smooth.
"""

from __future__ import annotations

# Resolve this checkout independently of the caller's working directory.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_ROOT = next(p for p in _RepoPath(__file__).resolve().parents
                  if (p / "EAP-IG").is_dir() and (p / "applications").is_dir())
_SUPPLEMENTAL_ROOT = _REPO_ROOT / "results" / "supplemental"
for _import_root in (_REPO_ROOT, _REPO_ROOT / "EAP-IG",
                     _REPO_ROOT / "EAP-IG" / "src", _REPO_ROOT / "data_gens"):
    if str(_import_root) not in _repo_sys.path:
        _repo_sys.path.insert(0, str(_import_root))


import argparse
import json
from typing import Dict, Mapping

PROFILES = ("easy", "hard", "helpful")
DEFAULT_START = {"easy": 0.60, "hard": 0.15, "helpful": 0.25}
DEFAULT_END = {"easy": 0.10, "hard": 0.60, "helpful": 0.30}


def parse_ratios(raw: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for piece in raw.split(","):
        key, sep, value = piece.strip().partition("=")
        if not sep:
            raise ValueError(f"expected key=value, got {piece!r}")
        if key not in PROFILES:
            raise ValueError(f"unsupported profile {key!r}; expected {PROFILES}")
        values[key] = float(value)
    if set(values) != set(PROFILES):
        raise ValueError(f"ratios must define exactly {PROFILES}, got {sorted(values)}")
    if any(value < 0 for value in values.values()):
        raise ValueError(f"ratios must be non-negative: {values}")
    total = sum(values.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {total}: {values}")
    return values


def interp_ratios(
    t: float,
    start: Mapping[str, float] = DEFAULT_START,
    end: Mapping[str, float] = DEFAULT_END,
) -> Dict[str, float]:
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"t must be in [0,1], got {t}")
    return {
        profile: float(start[profile] + (end[profile] - start[profile]) * t)
        for profile in PROFILES
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Print batch-level linear schedule targets")
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--start", default="easy=0.60,hard=0.15,helpful=0.25")
    parser.add_argument("--end", default="easy=0.10,hard=0.60,helpful=0.30")
    args = parser.parse_args()
    if args.num_batches <= 0:
        raise ValueError("num-batches must be positive")
    start = parse_ratios(args.start)
    end = parse_ratios(args.end)
    rows = []
    for batch_idx in range(args.num_batches):
        t = batch_idx / max(1, args.num_batches - 1)
        rows.append({"batch_index": batch_idx, "t": t, "target": interp_ratios(t, start, end)})
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
