# TemporalLoRA

TemporalLoRA is a parameter-efficient temporal adaptation approach for surgical VideoVQA.
It injects temporal mixing into the low-rank adaptation path of the vision encoder, targeting better temporal reasoning on endoscopic video clips.

This repository includes:
- Training and evaluation entrypoints for `Qwen3-VL` and `InternVL3`
- Temporal adapter modules and adapter installers
- REAL-Colon-VQA annotation files (in-domain and out-of-domain variants)

## Method Figure

![TemporalLoRA architecture](figs/temporallora_schema.png)

## Repository Layout

```text
.
|-- datasets/
|   `-- real_colon_vqa/
|       |-- real_colon_vqa_in_domain.jsonl
|       |-- real_colon_vqa_out_of_domain.jsonl
|       |-- video_split_by_procedure.json
|       |-- annotation_report.json
|       |-- yes_no_distribution.json
|       |-- train/
|       |-- val/
|       `-- test/
|-- figs/
|-- scripts/
|   |-- run_qwen3vl.py
|   |-- run_internvl3.py
|   |-- training.py
|   |-- temporal_modules.py
|   |-- adapters_qwen.py
|   |-- adapters_intern.py
|   |-- system_prompt_real_colon.py
|   `-- system_prompt_endovis.py
`-- README.md
```

## Setup

Use Python 3.10+ and install the required libraries:

```bash
pip install torch torchvision "transformers>=4.57.0" qwen-vl-utils peft evaluate numpy
```

Depending on your hardware/runtime, you may also need:
- `bitsandbytes` for 4-bit/8-bit loading
- `accelerate` for distributed or optimized execution

## Dataset Naming

REAL-Colon-VQA files are organized as:
- `real_colon_vqa_in_domain.jsonl`: in-template questions
- `real_colon_vqa_out_of_domain.jsonl`: out-of-template questions
- split folders (`train/`, `val/`, `test/`) containing long/short variants and `video_ids.txt`

## External Frame/Data Layout

The training scripts load annotations from this repository and expect frame data in external folders.

REAL-Colon-VQA frames (passed via `--real_frames_root`):

```text
<real_frames_root>/
|-- 002-001/
|-- 002-002/
|-- ...
`-- 002-012/
```

Frame names must match the identifiers in `frames` fields from annotation files (for example `002-006_6718`), with the image extension expected by the data loader.

EndoVis root (passed via `--endovis_root`) with default `--endovis_tail vqa/Sentence`:

```text
<endovis_root>/
|-- seq_1/
|   |-- left_frames/
|   `-- vqa/
|       `-- Sentence/
|-- seq_2/
|   |-- left_frames/
|   `-- vqa/
|       `-- Sentence/
`-- ...
```

For out-of-template EndoVis evaluation, provide a second root with the same sequence structure via `--endovis_ood_root`.

## Run Training

From the repository root:

```bash
python scripts/run_qwen3vl.py --help
python scripts/run_internvl3.py --help
```

Example commands:

```bash
python scripts/run_qwen3vl.py --epochs 20 --peft_enc stlora_mhatt_enc --peft_dec dora
python scripts/run_internvl3.py --epochs 20 --peft_enc stlora_mhatt_enc --peft_dec dora
```

## Paper-Style Testing

Based on the paper, key settings are:
- 2 backbones: `Qwen3-VL` and `InternVL3`
- in-template and out-of-template evaluation
- comparison across PEFT variants
- 20 epochs for fine-tuning runs

The scripts already expose this via `--peft_enc`, `--peft_dec`, `--train_endovis`, and evaluation flags.

### 1) Set data roots (PowerShell)

```powershell
$REAL_FRAMES = "D:\data\frames_resized"
$ENDOVIS_ROOT = "D:\data\EndoVis-18-VQA"
$ENDOVIS_OOD = "D:\data\Endovis18-VQA_Out-of-Template"
```

### 2) Zero-shot evaluation

REAL-Colon-VQA (in-template + OOD):

```powershell
python scripts/run_qwen3vl.py `
  --zero_shot_eval `
  --real_frames_root $REAL_FRAMES `
  --max_samples_eval 200

python scripts/run_internvl3.py `
  --zero_shot_eval `
  --real_frames_root $REAL_FRAMES `
  --max_samples_eval 200
```

EndoVis (in-template + OOD):

```powershell
python scripts/run_qwen3vl.py `
  --zero_shot_eval `
  --train_endovis `
  --endovis_root $ENDOVIS_ROOT `
  --endovis_ood_root $ENDOVIS_OOD `
  --sequence_length 8 `
  --max_samples_eval 200

python scripts/run_internvl3.py `
  --zero_shot_eval `
  --train_endovis `
  --endovis_root $ENDOVIS_ROOT `
  --endovis_ood_root $ENDOVIS_OOD `
  --sequence_length 8 `
  --max_samples_eval 200
```

### 3) Adapter comparison runs (20 epochs)

Example adapter presets:
- LoRA: `--peft_enc lora_enc --peft_dec lora`
- DoRA: `--peft_enc dora_enc --peft_dec dora`
- VeRA: `--peft_enc vera_enc --peft_dec vera`
- AdaLoRA: `--peft_enc adalora_enc --peft_dec adalora`
- Temporal LoRA (MHA): `--peft_enc stlora_mhatt_enc --peft_dec lora`
- Temporal DoRA (MHA): `--peft_enc stdora_mhatt_enc --peft_dec dora`

REAL-Colon-VQA, Qwen3-VL example:

```powershell
$env:TRAIN_OUT_SUFFIX = "_stdora_mhatt_real"
python scripts/run_qwen3vl.py `
  --epochs 20 `
  --peft_enc stdora_mhatt_enc `
  --peft_dec dora `
  --real_frames_root $REAL_FRAMES
```

EndoVis, InternVL3 example:

```powershell
$env:TRAIN_OUT_SUFFIX = "_stdora_mhatt_endovis"
python scripts/run_internvl3.py `
  --epochs 20 `
  --train_endovis `
  --peft_enc stdora_mhatt_enc `
  --peft_dec dora `
  --endovis_root $ENDOVIS_ROOT `
  --endovis_ood_root $ENDOVIS_OOD `
  --sequence_length 8
```

For 5-seed runs:

```powershell
python scripts/run_qwen3vl.py ... --repeat_5 --seed 42
python scripts/run_internvl3.py ... --repeat_5 --seed 42
```

### 4) Where test outputs are saved

Each run writes under `outputs/` with an auto-generated run folder.

Common files:
- `best_model.pt`, `last_model.pt`
- `metrics_real_test.json`
- `metrics_real_ood_test.json`
- `metrics/metrics_endovis.json`
- `metrics/metrics_endovis_ood.json`
- `metrics/predictions_*.json`

## Notes

- The training scripts expect external resources such as frame directories and EndoVis data paths; pass them with CLI flags when needed.
- Output checkpoints and logs are written under `outputs/`.
