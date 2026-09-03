# MedPID - Medical Prompt Injection Detection

MedPID is the medical prompt-injection detector used by MedPIGuard. It detects malicious instructions in doctor-patient dialogue turns using BiomedBERT pair embeddings and lightweight neural classifiers.

## Pipeline

```text
Doctor question ----\
                    +--> BiomedBERT --> [hq; ha; |hq-ha|; hq*ha; cos] --> classifier --> benign / attack
Patient answer  ----/
```

The fused feature vector has `4 * hidden_dim + 1` dimensions. With BiomedBERT base embeddings, this is 3073 dimensions.

## Models

`main.py` supports these model keys:

| Key | Model |
| --- | --- |
| `mlp` | Multi-layer perceptron classifier |
| `lstm_mlp` | LSTM over feature chunks with MLP head |
| `bilstm` | BiLSTM classifier |
| `autoencoder` | Unsupervised anomaly detector trained on benign samples |
| `transformer_encoder` | Transformer encoder classifier |
| `tabnet` | TabNet-style classifier |
| `attention_mlp` | Attention-MLP classifier |
| `all` | Train/evaluate all supported models |

## Data

The default datasets are stored in `data/`:

```text
data/
|-- trainset.csv
|-- testset.csv
|-- train_dialogue_turns_AgentClinic.json
|-- train_dialogue_turns_MedAgentSim.json
|-- test_dialogue_turns.json
|-- test_dialogue_turns_AgentClinic.json
`-- test_dialogue_turns_MedAgentSim.json
```

CSV files are used by MedPID training. JSON dialogue-turn files are used by the root detector benchmark.

## Quick Start

Run commands from this directory:

```bash
cd medpid
```

Train and evaluate all models:

```bash
PYTHONPATH=. python main.py
```

Train one model:

```bash
PYTHONPATH=. python main.py --model attention_mlp
```

Evaluate saved checkpoints without retraining:

```bash
PYTHONPATH=. python main.py --test-only
```

Use custom data or output paths:

```bash
PYTHONPATH=. python main.py \
  --train_csv data/trainset.csv \
  --test_csv data/testset.csv \
  --output_dir results/models_output
```

Run embedding baseline experiments:

```bash
PYTHONPATH=. python scripts/baseline_embedding_models.py --section all
```

Run only small-LLM embedding encoders:

```bash
PYTHONPATH=. python scripts/baseline_embedding_models.py --section embeddings --category llm_embed
```

## Project Structure

```text
medpid/
|-- main.py
|-- data/
|-- scripts/
|   |-- DialogueTurnExtraction2trainset.py
|   |-- DialogueTurnExtraction2testset.py
|   |-- baseline_embedding_models.py
|   |-- eval_extracted_turns.py
|   `-- run_baseline.sh
|-- src/
|   |-- config/
|   |-- data/
|   |-- models/
|   |-- utils/
|   `-- train.py
`-- results/
    `-- models_output/       # JSON summaries are trackable; large checkpoints are ignored
```

## Generated Artifacts

Large generated artifacts such as checkpoints (`*.pt`), cached embeddings (`*.npy`, `*.npz`), scalers (`*.pkl`), logs, and archives are ignored by the root `.gitignore`. Regenerate them locally or distribute them outside the Git repository.

## Notes

Full training and evaluation may download Hugging Face models and benefits from a CUDA GPU. See the root [README.md](../README.md) for the full MedPIGuard benchmark workflow and citation.
