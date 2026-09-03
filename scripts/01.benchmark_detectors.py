"""
scripts/01.benchmark_detectors.py

Pipeline đánh giá detector end-to-end với 1 lệnh:
    1. Đọc file dialogue turns đầu vào
    2. Prepare dữ liệu trong RAM (không lưu file trung gian)
    3. Chạy các detector baseline
    4. Xuất 1 file kết quả tổng hợp cuối cùng

Cách dùng:
        python scripts/01.benchmark_detectors.py \\
                --input MedPIDetector/data/test_dialogue_turns.json \\
                --output results/baseline_results_test_json.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemes.defenses import (  # noqa: E402
    CommandSansDetector,
    InstructionDetector,
    MedPIDDetector,
    ProtectAIDetector,
    SentinelDetector,
    SentinelV2Detector,
)


DetectorFactory = Callable[[], object]


# --------------------------------------------------------------------------- #
# STEP 1: PREPARE DATA
# --------------------------------------------------------------------------- #

def prepare_data_for_baseline(
    input_file: Path,
    output_file: Path | None = None,
) -> List[Dict[str, object]]:
    """Extract text from dialogue turns for baseline evaluation.

    The input has structure:
    - doctor: doctor's message
    - patient: patient's message (contains potential jailbreak attempt)
    - label: 0 (benign) or 1 (malicious)

    Output format compatible with evaluate_baseline:
    - text: patient_answer field
    - label: same as input
    """
    with open(input_file, "r", encoding="utf-8") as f:
        dialogues = json.load(f)

    prepared_data = []
    for dialogue in dialogues:
        prepared_data.append({
            "text": dialogue.get("patient_answer", ""),
            "label": dialogue.get("label", 0),
        })

    print(f"Prepared {len(prepared_data)} samples")

    if output_file is not None:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(prepared_data, f, indent=2, ensure_ascii=False)
        print(f"Output saved to: {output_file}")
    else:
        print("Skip saving prepared data file (in-memory mode).")

    label_counts: Dict[object, int] = {}
    for item in prepared_data:
        label = item["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  Label {label}: {count} samples ({count / len(prepared_data) * 100:.1f}%)")

    return prepared_data


# --------------------------------------------------------------------------- #
# STEP 2: EVALUATE DETECTORS
# --------------------------------------------------------------------------- #

def normalize_prediction(output: Dict[str, object]) -> int:
    label = str(output.get("label", "")).strip().lower()
    score = float(output.get("score", 0.5))

    positive = {
        "jailbreak", "instruction_injected", "inject", "injection",
        "malicious", "prompt_injection", "label_1", "1",
    }
    negative = {
        "benign", "clean", "safe", "not_injection", "label_0", "0",
    }

    if label in positive:
        return 1
    if label in negative:
        return 0

    if any(token in label for token in ["jail", "inject", "mal", "harm", "attack"]) or "1" in label:
        return 1
    if any(token in label for token in ["benign", "safe", "clean"]) or "0" in label:
        return 0

    return 1 if score >= 0.5 else 0


def resolve_device(device_arg: str) -> str | None:
    if device_arg == "auto":
        return None
    return device_arg


def _resolve_commandsans_model(commandsans_model: str | None) -> str | None:
    """Resolve đường dẫn model CommandSans: ưu tiên argument, sau đó fallback model local."""
    if commandsans_model is not None:
        return commandsans_model
    default_path = PROJECT_ROOT / "schemes" / "CommandSans" / "release" / "commandsans-final"
    if default_path.exists():
        return str(default_path)
    return None


def build_detector_factories(
    device: str | None,
    instruction_device: str | None,
    commandsans_model: str | None,
    medpid_model: str | None,
    medpid_threshold: float | None,
    device_ids: list[int] | None = None,
    instruction_num_devices: int = 1,
) -> Dict[str, DetectorFactory]:
    """Xây dựng factories detector với phân phối GPU."""
    if device_ids is None:
        device_ids = [0] if device == "auto" else [0]

    factories: Dict[str, DetectorFactory] = {}

    sentinel_device = (
        f"cuda:{device_ids[0]}" if device == "auto"
        else (device if "cuda" in str(device) else "cpu")
    )
    factories["protectai"] = lambda: ProtectAIDetector(device=sentinel_device)
    factories["sentinel"] = lambda: SentinelDetector(device=sentinel_device)
    factories["sentinelv2"] = lambda: SentinelV2Detector(device=sentinel_device)

    instruction_device_arg = instruction_device if instruction_device is not None else sentinel_device
    factories["Instruction"] = lambda: InstructionDetector(
        device=instruction_device_arg,
        device_ids=instruction_num_devices,
    )

    resolved_model = _resolve_commandsans_model(commandsans_model)
    factories["commandsans"] = lambda: CommandSansDetector(
        model_name_or_path=resolved_model,
        device=sentinel_device,
    )

    medpid_device = f"cuda:{device_ids[-1]}" if "cuda" in str(sentinel_device) else None
    factories["medpid"] = lambda: MedPIDDetector(
        model_path=medpid_model,
        device=medpid_device,
        threshold=medpid_threshold,
    )

    return factories


def evaluate_detector(name: str, detector: object, texts: List[str], labels: List[int]) -> Dict[str, object]:
    import time

    predictions: List[int] = []
    latencies: List[float] = []

    start_time = time.time()
    for i, text in enumerate(tqdm(texts, desc=f"Evaluating {name}", unit="sample")):
        output = detector.predict(text)
        predictions.append(normalize_prediction(output))
        latency = (time.time() - start_time) * 1000 / (i + 1)
        latencies.append(latency)

    end_time = time.time()
    total_time = end_time - start_time
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0

    accuracy = float(accuracy_score(labels, predictions))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).tolist()

    return {
        "name": name,
        "num_samples": len(texts),
        "total_time_seconds": round(total_time, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "min_latency_ms": round(min_latency, 2),
        "max_latency_ms": round(max_latency, 2),
        "throughput_samples_per_sec": round(len(texts) / total_time, 2) if total_time > 0 else 0,
        "accuracy": accuracy,
        "precision_pos1": float(precision),
        "recall_pos1": float(recall),
        "f1_pos1": float(f1),
        "confusion_matrix_labels_0_1": matrix,
    }


def _rows_to_texts_labels(rows: List[Dict[str, object]]) -> Tuple[List[str], List[int], List[Dict[str, object]]]:
    texts: List[str] = []
    labels: List[int] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i} is not a JSON object.")
        if "text" not in row or "label" not in row:
            raise ValueError(f"Row {i} must contain 'text' and 'label'.")
        texts.append(str(row["text"]))
        labels.append(int(row["label"]))

    return texts, labels, rows


def load_dataset(path: Path, max_samples: int | None) -> Tuple[List[str], List[int], List[Dict[str, object]]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise ValueError("Dataset JSON must be a list of records.")

    if max_samples is not None:
        rows = rows[:max_samples]

    return _rows_to_texts_labels(rows)


def save_detector_result(
    output_dir: Path,
    detector_name: str,
    metrics: Dict[str, object],
    dataset_path: str,
    num_samples: int,
) -> None:
    """Lưu kết quả của một detector vào file JSON riêng."""
    result = {
        "dataset": dataset_path,
        "num_samples": num_samples,
        "detector": detector_name,
        "metrics": metrics,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    file_path = output_dir / f"{detector_name}_result.json"
    file_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved: {file_path}")


def merge_detector_results(output_dir: Path, detectors: List[str]) -> Dict[str, object]:
    """Gộp kết quả từ các file detector riêng lẻ thành báo cáo tổng hợp."""
    report: Dict[str, object] = {"results": [], "errors": []}
    for name in detectors:
        file_path = output_dir / f"{name}_result.json"
        if file_path.exists():
            data = json.loads(file_path.read_text(encoding="utf-8"))
            report["results"].append(data["metrics"])
        else:
            report["errors"].append({"name": name, "error": "No result file found"})
    return report


def run_evaluation(
    data: str,
    detectors: str,
    device: str,
    instruction_device: str | None,
    commandsans_model: str | None,
    medpid_model: str | None,
    medpid_threshold: float | None,
    max_samples: int | None,
    output: str,
    fail_fast: bool,
    gpu_ids: List[int],
    instruction_gpus: int,
    prepared_rows: List[Dict[str, object]] | None = None,
    prepared_source: str | None = None,
) -> None:
    data_path: Path | None = None
    if prepared_rows is None:
        data_path = Path(data)
        if not data_path.is_absolute():
            data_path = Path.cwd() / data_path
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if prepared_rows is None:
        texts, labels, _ = load_dataset(data_path, max_samples)
        dataset_name = str(data_path)
    else:
        rows = prepared_rows[:max_samples] if max_samples is not None else prepared_rows
        texts, labels, _ = _rows_to_texts_labels(rows)
        dataset_name = prepared_source if prepared_source is not None else "in-memory-prepared-data"

    selected = [key.strip() for key in detectors.split(",") if key.strip()]
    if not selected:
        raise ValueError("No detectors selected. Use --detectors.")

    resolved_device = resolve_device(device)
    factories = build_detector_factories(
        device=resolved_device,
        instruction_device=instruction_device,
        commandsans_model=commandsans_model,
        medpid_model=medpid_model,
        medpid_threshold=medpid_threshold,
        device_ids=gpu_ids,
        instruction_num_devices=instruction_gpus,
    )

    unknown = sorted(set(selected) - set(factories.keys()))
    if unknown:
        raise ValueError(f"Unknown detectors: {unknown}. Available: {sorted(factories.keys())}")

    report: Dict[str, object] = {
        "dataset": dataset_name,
        "num_samples": len(texts),
        "settings": {
            "detectors": selected,
            "device": device,
            "instruction_device": instruction_device,
            "max_samples": max_samples,
        },
        "results": [],
        "errors": [],
    }

    for key in selected:
        try:
            print(f"Running {key}...")
            detector = factories[key]()
            metrics = evaluate_detector(key, detector, texts, labels)

            report["results"].append(metrics)
            print(f"Done {key}: acc={metrics['accuracy']:.4f}, f1_pos1={metrics['f1_pos1']:.4f}")
        except Exception as exc:  # pragma: no cover
            err = {"name": key, "error": str(exc)}
            report["errors"].append(err)
            print(f"Failed {key}: {exc}")
            if fail_fast:
                break

    report["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Báo cáo tổng hợp: {output_path} ({report['timestamp']})")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=str, default="medpid/data/test_dialogue_turns.json",
                         help="Path to extracted_dialogue_turns.json")
    parser.add_argument("--detectors", type=str, default="protectai,sentinel,sentinelv2,Instruction",
                         help="Comma-separated detector keys: protectai,sentinel,sentinelv2,Instruction,commandsans,medpid")
    parser.add_argument("--device", type=str, default="cuda",
                         help="Device for HF detectors: auto, cpu, cuda, cuda:0...")
    parser.add_argument("--instruction-device", type=str, default=None,
                         help="Optional dedicated device for InstructionDetector (e.g. cpu)")
    parser.add_argument("--commandsans-model", type=str, default=None,
                         help="Fine-tuned CommandSans checkpoint path or Hugging Face model ID.")
    parser.add_argument("--medpid-model", type=str, default=None,
                         help="MedPID checkpoint path (.pt). If None, auto-selects best model (Attention-MLP).")
    parser.add_argument("--medpid-threshold", type=float, default=None,
                         help="Override classification threshold for MedPID (default: from checkpoint).")
    parser.add_argument("--max-samples", type=int, default=None,
                         help="Optional cap on number of samples to evaluate")
    parser.add_argument("--output", type=str, default="results/baseline_results_test_json.json",
                         help="Output JSON report path")
    parser.add_argument("--fail-fast", action="store_true",
                         help="Stop immediately when any detector fails")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0],
                         help="GPU IDs to use for detectors (e.g. --gpu-ids 0 1 cho 2 GPU)")
    parser.add_argument("--instruction-gpus", type=int, default=1,
                         help="Số GPU để chia Instruction Detector (mặc định 1, dùng --instruction-gpus 2 để chia lên 2 GPU)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MedPID end-to-end benchmark pipeline (single command)"
    )
    add_pipeline_args(parser)

    args = parser.parse_args()
    prepared_rows = prepare_data_for_baseline(Path(args.input), None)
    run_evaluation(
        data=args.input,
        detectors=args.detectors,
        device=args.device,
        instruction_device=args.instruction_device,
        commandsans_model=args.commandsans_model,
        medpid_model=args.medpid_model,
        medpid_threshold=args.medpid_threshold,
        max_samples=args.max_samples,
        output=args.output,
        fail_fast=args.fail_fast,
        gpu_ids=args.gpu_ids,
        instruction_gpus=args.instruction_gpus,
        prepared_rows=prepared_rows,
        prepared_source=args.input,
    )


if __name__ == "__main__":
    main()