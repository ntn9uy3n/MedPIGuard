# MedPIGuard

Official source code for the paper: **"MedPIGuard: Securing Multi-Agent Clinical AI Against Prompt Injection Attacks"**.

MedPIGuard is a research codebase for studying prompt injection attacks and defenses in multi-agent clinical AI systems. The repository contains clinical agent simulators, attack experiment artifacts, defense wrappers, detector benchmarks, and MedPID, a medical prompt-injection detector built for doctor-patient dialogue turns.

## Overview

Clinical multi-agent AI systems exchange patient messages, notes, tool outputs, and intermediate responses across agents. Malicious instructions embedded in those channels can override clinical goals, manipulate downstream agents, or leak protected system behavior.

This repository supports the paper experiments through:

- attack-effectiveness experiments on clinical multi-agent systems;
- attack-success-rate comparisons across prompt-injection defenses;
- detector benchmarking for general and medical prompt-injection detectors;
- MedPID, a BiomedBERT-based detector for medical dialogue turns.

## Repository Layout

```text
MedPIGuard/
|-- scripts/
|   |-- 01.benchmark_detectors.py      # Detector benchmark entry point
|   `-- judges_finall.py               # LLM judge utilities for experiment outputs
|-- schemes/
|   |-- defenses.py                    # Detector wrappers and MedPID integration
|   `-- medpiguard.py                  # MedPIGuard defense utilities
|-- medpid/
|   |-- main.py                        # Train and evaluate MedPID models
|   |-- data/                          # Processed dialogue-turn datasets
|   |-- scripts/                       # Data extraction and baseline scripts
|   |-- src/                           # MedPID model, data, and training code
|   `-- README.md                      # MedPID-specific documentation
|-- systems/
|   |-- AgentClinic/                   # AgentClinic-based clinical simulator code
|   `-- MedAgentSim/                   # MedAgentSim-based clinical simulator code
|-- experiments/
|   |-- AttackEffectiveness/           # Attack effectiveness artifacts
|   `-- AttackSuccessRateComparison/   # Defense comparison artifacts
|-- results/                           # Benchmark result JSON files
`-- requirements.txt
```

## Installation

Python 3.10+ is recommended. A CUDA GPU is recommended for full reproduction, while CPU can be used for small checks.

```bash
git clone https://github.com/ntn9uy3n/MedPIGuard.git
cd MedPIGuard

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some detectors download Hugging Face models on first use. If you use the `Instruction` detector, start the instruction-detection API service first or update the API URL in `schemes/defenses.py`.

## Data Format

Processed dialogue-turn datasets are stored under `medpid/data/`. JSON records use the following fields:

```json
{
  "doctor_question": "...",
  "patient_answer": "...",
  "label": 0
}
```

`label = 0` denotes benign text. `label = 1` denotes prompt injection or malicious instruction text.

## Quick Start

Run a small detector benchmark on CPU:

```bash
python scripts/01.benchmark_detectors.py \
  --input medpid/data/test_dialogue_turns.json \
  --output results/quick_benchmark.json \
  --detectors protectai,sentinel,sentinelv2 \
  --device cuda \
  --max-samples 100
```

Run the benchmark with MedPID after training or placing a supported checkpoint under `medpid/results/models_output/`:

```bash
python scripts/01.benchmark_detectors.py \
  --input medpid/data/test_dialogue_turns.json \
  --output results/medpid_benchmark.json \
  --detectors medpid \
  --device cuda
```

Train and evaluate MedPID:

```bash
cd medpid
PYTHONPATH=. python main.py
```

Evaluate an existing MedPID checkpoint:

```bash
cd medpid
PYTHONPATH=. python main.py --test-only
```

Run embedding baseline experiments:

```bash
cd medpid
PYTHONPATH=. python scripts/baseline_embedding_models.py --section all
```

More details for MedPID are available in [medpid/README.md](medpid/README.md).

## Supported Detectors

The benchmark pipeline supports these detector keys:

| Key | Detector |
| --- | --- |
| `protectai` | `ProtectAI/deberta-v3-base-prompt-injection-v2` |
| `sentinel` | `qualifire/prompt-injection-sentinel` |
| `sentinelv2` | `rogue-security/prompt-injection-jailbreak-sentinel-v2` |
| `Instruction` | Instruction-injection detection API wrapper |
| `commandsans` | Optional local CommandSans sanitizer/checkpoint |
| `medpid` | MedPID medical prompt-injection detector |

## Reproducing Experiments

1. Install dependencies with `pip install -r requirements.txt`.
2. Inspect or regenerate simulator artifacts in `systems/` and `experiments/`.
3. Extract dialogue turns with the scripts in `medpid/scripts/` if regenerating datasets from simulator outputs.
4. Train MedPID with `cd medpid && PYTHONPATH=. python main.py`.
5. Benchmark detectors with `python scripts/01.benchmark_detectors.py`.

Full reproduction may require external model downloads, GPU resources, and local API services for optional detectors.

## Repository Hygiene

The repository is configured to ignore local/private and generated artifacts, including:

- `.env` files and virtual environments;
- `.zip` and other archive exports;
- logs, temporary folders, Python caches, and test caches;
- large model/checkpoint artifacts such as `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.npy`, `*.npz`, and `*.pkl`.

Large model artifacts should be regenerated locally or distributed through release assets/external storage rather than committed to GitHub.

## Citation

If you use this code, please cite:

```bibtex
@article{medpiguard,
  title = {MedPIGuard: Securing Multi-Agent Clinical AI Against Prompt Injection Attacks},
  author = {Nguyen, Ngoc-Truong and Le, Kim-Hung},
  year = {2026}
}
```

Update the BibTeX entry with the final venue and author list after publication.

## License

This repository contains source code and experiment artifacts for academic research. Third-party simulator components under `systems/AgentClinic/` and `systems/MedAgentSim/` may have their own licenses.
