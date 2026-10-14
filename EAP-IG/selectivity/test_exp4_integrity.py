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


import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


from selectivity import build_selectivity_table as matrix
from selectivity import run_registry as registry
from selectivity import spearman_and_jaccard as correl
from selectivity import trajectory_with_integrity as trajectory


class Exp4IntegrityTests(unittest.TestCase):
    def test_registry_contract_hashes_input_and_effective_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            input_path.write_text('{"version":1}\n')
            meta = {
                "dataset": "sciq",
                "score_key": "aum",
                "model_name": "model",
                "dtype": "bfloat16",
                "device": "cuda",
                "seed": 42,
                "epochs": 5,
                "batch_size": 4,
                "grad_accum": 1,
                "el2n_epochs": 2,
                "max_samples": 0,
                "max_length": 512,
                "topk_edges": 250,
                "topk_rank_abs": True,
                "input_all_scores_json": str(input_path),
                "input_all_scores_json_sha256": registry.file_sha256(str(input_path)),
                "resume_state_dict": "",
                "resume_state_dict_sha256": "",
                "lr": 2e-5,
                "weight_decay": 0.0,
                "ablation_enabled": False,
                "core_trainer": str(registry.CORE_TRAINER.resolve()),
                "core_trainer_sha256": registry.file_sha256(str(registry.CORE_TRAINER)),
                "integrity_wrapper": str(registry.INTEGRITY_WRAPPER.resolve()),
                "integrity_wrapper_sha256": registry.file_sha256(str(registry.INTEGRITY_WRAPPER)),
            }
            kwargs = dict(
                score_key="aum", run_kind="full", topk=250, seed=42,
                input_json=input_path, model_name="model", dtype="bfloat16",
                device="cuda", epochs=5, batch_size=4, grad_accum=1,
                lr=2e-5, weight_decay=0.0, max_samples=0, max_length=512,
                resume_state_dict="", el2n_epochs=2, random_edges_seed=42,
                circuit_bucket="high", corrupt_type="gaussian",
            )
            self.assertTrue(registry.config_matches(meta, **kwargs))
            tampered_code = dict(meta)
            tampered_code["integrity_wrapper_sha256"] = "0" * 64
            self.assertFalse(registry.config_matches(tampered_code, **kwargs))
            self.assertFalse(registry.config_matches(meta, **{**kwargs, "el2n_epochs": 999}))
            self.assertFalse(
                registry.config_matches(
                    meta, **{**kwargs, "resume_state_dict": "/different/init.pt"}
                )
            )
            input_path.write_text('{"version":2}\n')
            self.assertFalse(registry.config_matches(meta, **kwargs))

    def test_cross_activation_request_digests_must_match(self) -> None:
        good = {
            metric: {"utility_prompt_sha256": "prompt", "hf_id_sha256": "ids"}
            for metric in correl.UTILITY_KEYS
        }
        self.assertEqual(correl.validate_cross_request_identity(good), ("prompt", "ids"))
        bad = {key: dict(value) for key, value in good.items()}
        bad["grad_align"]["utility_prompt_sha256"] = "different"
        with self.assertRaises(RuntimeError):
            correl.validate_cross_request_identity(bad)

    def test_source_prompt_gold_digest_detects_per_hf_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write(name: str, prompt: str) -> Path:
                path = root / name
                path.write_text(json.dumps({"items": [{
                    "hf_id": "sciq/train/0", "prompt": prompt,
                    "gold_token_id": 1, "gold_label": "A",
                    "label_space": ["A", "B"],
                    "label_token_ids": {"A": 1, "B": 2},
                }]}))
                return path
            first = write("first.json", "same")
            second = write("second.json", "same")
            third = write("third.json", "changed")
            first_digest = matrix._source_prompt_gold_digest(str(first.resolve()))
            self.assertEqual(first_digest, matrix._source_prompt_gold_digest(str(second.resolve())))
            self.assertNotEqual(first_digest, matrix._source_prompt_gold_digest(str(third.resolve())))

    def test_selectivity_is_literal_signed_ratio_with_zero_policy(self) -> None:
        self.assertEqual(matrix.selectivity_index([2.0, -1.0, -3.0], 0), -1.0)
        self.assertTrue(math.isnan(matrix.selectivity_index([2.0, 1.0, -1.0], 0)))

    def test_exp4_wrapper_attaches_content_and_code_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            input_path.write_text("input-v1")
            out_root = root / "runs"

            def fake_trainer(_command):
                run_dir = out_root / "sciq" / "aum" / "synthetic"
                checkpoint_dir = run_dir / "checkpoints"
                checkpoint_dir.mkdir(parents=True)
                checkpoint = checkpoint_dir / "final.pt"
                checkpoint.write_bytes(b"checkpoint-v1")
                scores = run_dir / "all_scores_sciq.json"
                scores.write_text(json.dumps({
                    "meta": {
                        "input_all_scores_json": str(input_path),
                        "circuit_pt": "",
                    },
                    "items": [],
                }))
                (run_dir / "run_summary.json").write_text(json.dumps({
                    "run_dir": str(run_dir),
                    "all_scores_json": str(scores),
                    "final_checkpoint": str(checkpoint),
                    "meta": {},
                }))
                return mock.Mock(returncode=0)

            with mock.patch.object(trajectory.subprocess, "run", side_effect=fake_trainer):
                trajectory.main([
                    "--dataset", "sciq", "--score_key", "aum",
                    "--all_scores_json", str(input_path),
                    "--out_root", str(out_root),
                    "--circuits_root", str(root),
                    "--disable_ablation", "--el2n_epochs", "3",
                ])

            summary_path = next(out_root.rglob("run_summary.json"))
            summary = json.loads(summary_path.read_text())
            scores_path = Path(summary["all_scores_json"])
            meta = json.loads(scores_path.read_text())["meta"]
            self.assertEqual(meta["input_all_scores_json_sha256"], trajectory.sha256_file(input_path))
            self.assertEqual(meta["el2n_epochs"], 3)
            self.assertEqual(meta["core_trainer_sha256"], trajectory.sha256_file(trajectory.CORE_TRAINER))
            self.assertEqual(meta["integrity_wrapper_sha256"], trajectory.sha256_file(trajectory.WRAPPER_PATH))
            self.assertEqual(summary["final_checkpoint_sha256"], meta["final_checkpoint_sha256"])
            self.assertEqual(summary["all_scores_json_sha256"], trajectory.sha256_file(scores_path))

    def test_strict_matrix_failure_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = root / "full.json"
            full.write_text(json.dumps({
                "meta": {"ablation_enabled": False},
                "items": [{"hf_id": "a"}, {"hf_id": "b"}],
            }))
            result = root / "result"
            with self.assertRaises(RuntimeError):
                matrix.main([
                    "--full-json", str(full), "--expected-samples", "2",
                    "--result-dir", str(result), "--require-complete",
                ])
            self.assertFalse(result.exists())

    def test_strict_correl_failure_does_not_publish_jaccard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            edge_args = []
            for circuit_index, circuit in enumerate(correl.CIRCUITS):
                edge = root / f"{circuit}.jsonl"
                edge.write_text("\n".join(json.dumps({
                    "edge": f"edge-{circuit_index}-{i}",
                    "matrix_index": [circuit_index, i], "score": 2 - i,
                }) for i in range(2)) + "\n")
                edge_args.extend([f"--edge-{circuit.replace('_', '-')}", str(edge)])
            result = root / "result"
            missing = root / "missing.csv"
            with self.assertRaises(FileNotFoundError):
                correl.main([
                    *edge_args, "--topk-edges", "2", "--expected-samples", "2",
                    "--activation-aum", str(missing),
                    "--activation-el2n", str(missing),
                    "--activation-grad-align", str(missing),
                    "--result-dir", str(result),
                ])
            self.assertFalse(result.exists())


if __name__ == "__main__":
    unittest.main()
