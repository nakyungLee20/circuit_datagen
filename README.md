# Training Utility Aware Circuits and Circuit Steered Data Generation

Code for "Mechanistic Circuit Identification for Controllable Data Generation" (NeurIPS 2026).

Discover circuits associated with training utility, use them to generate targeted MCQA data, and schedule the resulting pools during fine-tuning.

## Setup

Use Python 3.11 and a CUDA-capable GPU. Run commands from the repository root:

```bash
python -m pip install -r requirements.txt
```

Tested dependency versions are in [environment-tested.txt](environment-tested.txt). The reference generator is `Qwen/Qwen2.5-1.5B-Instruct`; attention steering uses Qwen2 eager-attention hooks. The modified EAP library is installed with the requirements.

## Main pipeline

Follow the stages below. Circuit validation checks the discovered circuits; structure analysis then supplies steering targets for generation.

| Stage | Entry point | Input → output |
| --- | --- | --- |
| 1. Training-utility scoring | [app_score_collect.py](data_gens/app_score_collect.py) | MCQA data → AUM/EL2N/GradAlign scores and high/low buckets |
| 2. Paired inputs | [app_input_pairs.py](data_gens/app_input_pairs.py) | Buckets → clean/corrupt input pairs |
| 3. EAP-IG discovery | [scen1_eapig.py](EAP-IG/scen1_eapig.py) | Pairs → high/low circuits for each metric |
| 4a. Faithfulness | [faithfulness.py](EAP-IG/faithfulness.py) | Circuits + pairs → recovery/faithfulness scores |
| 4b. Causal intervention | [zero_ablation_finetune.py](EAP-IG/zero_ablation_finetune.py), [zero_ablation.py](EAP-IG/zero_ablation.py) | Full/ablated trajectories or static interventions → utility changes |
| 4c. Filtering validation | [filter_finetune.py](applications/filter_finetune.py), [filter_inference.py](applications/filter_inference.py) | Circuit-score-filtered subsets → fine-tuning and evaluation |
| 5. Structure analysis | [structure_analysis.py](structure/structure_analysis.py), [visualize_circuits.py](structure/visualize_circuits.py) | Circuits → structure summaries and top-edge tables |
| 6. Steering export | [export_circuit_steering_map.py](structure/export_circuit_steering_map.py) | Structure tables → generator-ready circuit CSVs |
| 7. Seed and calibration data | [seed_dataset.py](applications/seed_dataset.py) | Source data + utility scores → seed/calibration JSONL |
| 8. Circuit-steered generation | [circuit_generation.py](applications/circuit_generation.py) | Seeds + calibration + circuit CSVs → selected data pools |
| 9. SAMS fine-tuning | [curriculum_ft.py](applications/curriculum_ft.py) | Source data + generated pools → three-stage fine-tuning |
| 10. Evaluation | [curriculum_if.py](applications/curriculum_if.py) | Fine-tuned checkpoints → MCQA results |

Filtering validation uses [circuit_scores.py](applications/circuit_scores.py) and [filter_data.py](applications/filter_data.py) internally. Model/GQA loading is handled by [gqa_circuits.py](EAP-IG/gqa_circuits.py).

Each entry point exposes its arguments through `--help`, for example:

```bash
python data_gens/app_score_collect.py --help
python applications/circuit_generation.py --help
python applications/curriculum_ft.py --help
```

## Connecting stages

Pass explicit input/output paths between stages. Two hand-offs also depend on directory names:

- **Structure analysis → steering export:** run `structure_analysis.py` and `visualize_circuits.py` once per metric with `--compare-a <high circuits> --compare-b <low circuits> --label-a high --label-b low --tag <metric>_gaussian` (e.g. `--tag aum_gaussian`). `export_circuit_steering_map.py --structure-root results/structure` then finds `<metric>_gaussian/visualizations/{high,low}/` for all three metrics. Add `--faith-root results/validation/faithfulness` to include the `faithfulness.py` results.
- **Generation → SAMS:** save each recipe with `--output-dir results/circuit_generation/Qwen__Qwen2.5-1.5B-Instruct/<recipe>/<run>`. `curriculum_ft.py` loads the newest matching `<recipe>_selected.jsonl` there through its defaults `--generation-root results/circuit_generation --generator-model-key Qwen__Qwen2.5-1.5B-Instruct`.

## Settings for the figure

- **Scoring/discovery:** collect AUM and EL2N with `--score_group dynamic`, and GradAlign with `--score_group data_attr --score_key grad_align`. Discover both `high` and `low` buckets for `aum`, `el2n`, and `grad_align` (six circuits). Keep model, sample population, corruption and GQA settings consistent across dependent stages.
- **Calibration:** pass dynamic scores through `--scores-json` and GradAlign scores through `--grad-align-json` to `seed_dataset.py`.
- **Generation:** use recipes `easy` (learnable), `hard` (challenging), and `helpful` (well-aligned). Supply `--input-jsonl`, `--calibration-jsonl`, and `--edge-table-dir`; enable `--gen-mode both --circuit-target-mode discovered --use-activation-addition --use-attention-steering`.
- **SAMS:** use `--baseline-mode ours --curriculum-mode on`, point `--generation-root` to the generated pools, and explicitly set the three-profile ratios below. Stages follow a predefined epoch schedule.

| Stage / argument | Generated-pool ratios |
| --- | --- |
| Warm-up: `--stage1-ratios` | `easy=0.55,hard=0.15,helpful=0.30` |
| Transition: `--stage2-ratios` | `easy=0.25,hard=0.45,helpful=0.30` |
| Challenge: `--stage3-ratios` | `easy=0.10,hard=0.65,helpful=0.25` |

Set `--flat-generated-ratios easy=0.25,hard=0.45,helpful=0.30` as well to exclude the extra `medium` profile. With `--epochs 6`, each stage receives two epochs. Ratios apply within the generated portion; `--source-ratio` controls the source-data share.

Local artifacts, checkpoints, logs, notes and shell wrappers are excluded from Git. The main pipeline runs through the Python entry points above.

The bundled EAP code retains its [upstream license](EAP-IG/LICENSE).
