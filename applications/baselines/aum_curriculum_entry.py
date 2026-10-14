"""Opt-in curriculum_ft entrypoint that saves a distinct state every epoch.

The core trainer is imported unchanged. Its single validation call per epoch is
wrapped so AUM receives an ordered learning trajectory rather than potentially
duplicate ``best.pt``/``final.pt`` files.
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


import hashlib
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = _REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "applications") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "applications"))

from applications import curriculum_ft


def _arg_value(args, flag: str, default: str = "") -> str:
    for idx, arg in enumerate(args):
        if arg == flag and idx + 1 < len(args):
            return args[idx + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return default


def _evenly_spaced_indices(numel: int, sample_count: int, *, device) -> torch.Tensor:
    """Return deterministic in-bounds sample positions without float rounding."""
    if numel <= 0:
        raise ValueError(f"numel must be positive, got {numel}")
    if sample_count <= 0 or sample_count > numel:
        raise ValueError(
            f"sample_count must be in [1, {numel}], got {sample_count}"
        )
    if sample_count == 1:
        index_values = [0]
    else:
        last_index = numel - 1
        index_values = [
            (position * last_index) // (sample_count - 1)
            for position in range(sample_count)
        ]
    return torch.tensor(index_values, dtype=torch.long, device=device)


def parameter_sample_sha256(model) -> str:
    """Deterministic state fingerprint using samples from every parameter."""
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            tensor = parameter.detach().reshape(-1)
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(parameter.shape)).encode("ascii"))
            digest.update(str(parameter.dtype).encode("ascii"))
            if tensor.numel() == 0:
                continue
            sample_count = min(64, tensor.numel())
            indices = _evenly_spaced_indices(
                tensor.numel(),
                sample_count,
                device=tensor.device,
            )
            sample = tensor.index_select(0, indices).float().cpu().contiguous().numpy()
            digest.update(sample.tobytes())
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: aum_curriculum_entry.py /path/to/curriculum_ft.py [args ...]")
    trainer_path = Path(sys.argv[1]).expanduser().resolve()
    if trainer_path != Path(curriculum_ft.__file__).resolve():
        raise RuntimeError(f"unexpected trainer path: {trainer_path}")
    trainer_args = sys.argv[2:]
    model_name_or_path = _arg_value(trainer_args, "--model-name-or-path")

    checkpoint_dir_raw = os.environ.get("AUM_EPOCH_CHECKPOINT_DIR", "")
    if not checkpoint_dir_raw:
        raise RuntimeError("AUM_EPOCH_CHECKPOINT_DIR must be set")
    checkpoint_dir = Path(checkpoint_dir_raw).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_dir / "manifest.json"

    original_eval = curriculum_ft.eval_mcqa_metrics
    entries = []
    fingerprints = set()

    def eval_and_save(model, dataloader, device):
        metrics = original_eval(model, dataloader, device)
        epoch = len(entries) + 1
        fingerprint = parameter_sample_sha256(model)
        if fingerprint in fingerprints:
            raise RuntimeError(
                f"epoch {epoch} duplicates an earlier AUM model state ({fingerprint})"
            )
        fingerprints.add(fingerprint)
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        curriculum_ft.save_checkpoint(
            checkpoint_path,
            model,
            epoch=epoch,
            global_step=0,
            meta={"aum_epoch_state": True, "parameter_sample_sha256": fingerprint},
        )
        entry = {
            "epoch": epoch,
            "path": str(checkpoint_path),
            "parameter_sample_sha256": fingerprint,
            "validation": metrics,
        }
        entries.append(entry)
        manifest_path.write_text(
            json.dumps(
                {
                    "model_name_or_path": model_name_or_path,
                    "fingerprint_kind": "sha256(samples_from_every_named_parameter)",
                    "checkpoints": entries,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[AUM] saved distinct epoch state {epoch}: {checkpoint_path}", flush=True)
        return metrics

    curriculum_ft.eval_mcqa_metrics = eval_and_save
    curriculum_ft.main(trainer_args)
    if len(entries) < 2:
        raise RuntimeError(f"AUM reference produced only {len(entries)} epoch states")
    print(f"[AUM] trajectory complete: {len(entries)} distinct states; manifest={manifest_path}")


if __name__ == "__main__":
    main()
