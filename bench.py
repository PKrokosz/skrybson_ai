#!/usr/bin/env python3
"""Utility script for benchmarking faster-whisper models on local samples."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch is optional for CPU runs
    torch = None


@dataclass
class Sample:
    """Single benchmark sample description."""

    sample_id: str
    category: str
    description: str
    reference_text: str
    audio_path: Optional[Path]


DEFAULT_MODELS = ["small", "medium", "large-v3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("bench/manifest.json"),
        help="Ścieżka do manifestu opisującego próbki.",
    )
    parser.add_argument(
        "--precomputed",
        type=Path,
        default=Path("bench/results/precomputed.json"),
        help="Ścieżka do pliku z zapisanymi wynikami modeli.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bench/results/latest_metrics.json"),
        help="Plik wyjściowy z podsumowaniem benchmarku.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Lista modeli do uruchomienia (domyślnie small, medium, large-v3).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Urządzenie przekazywane do faster-whisper (np. cuda lub cpu).",
    )
    parser.add_argument(
        "--compute-type",
        dest="compute_type",
        default="auto",
        help="Typ obliczeń faster-whisper (np. float16, int8_float16).",
    )
    parser.add_argument(
        "--run-models",
        action="store_true",
        help="Wymuś ponowne uruchomienie modeli zamiast korzystania z danych precomputed.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> Dict[str, Sample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping: Dict[str, Sample] = {}
    for entry in payload.get("samples", []):
        sample = Sample(
            sample_id=entry["id"],
            category=entry["category"],
            description=entry["description"],
            reference_text=entry["reference_text"],
            audio_path=Path(entry["audio_path"]) if entry.get("audio_path") else None,
        )
        mapping[sample.sample_id] = sample
    if not mapping:
        raise ValueError(f"Manifest {path} nie zawiera żadnych próbek")
    return mapping


def load_precomputed(path: Path) -> Dict[str, Dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    models: Dict[str, Dict[str, dict]] = {}
    for model in payload.get("models", []):
        runs = {}
        for run in model.get("runs", []):
            runs[run["sample_id"]] = run
        models[model["name"]] = {
            "device": model.get("device"),
            "compute_type": model.get("compute_type"),
            "runs": runs,
        }
    if not models:
        raise ValueError(f"Plik {path} nie zawiera danych o modelach")
    return models


def normalize_words(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def normalize_chars(text: str) -> List[str]:
    return list("".join(normalize_words(text)))


def levenshtein(ref: Iterable[str], hyp: Iterable[str]) -> int:
    ref_list = list(ref)
    hyp_list = list(hyp)
    if not ref_list:
        return len(hyp_list)
    if not hyp_list:
        return len(ref_list)
    prev_row = list(range(len(hyp_list) + 1))
    for i, ref_item in enumerate(ref_list, start=1):
        curr_row = [i]
        for j, hyp_item in enumerate(hyp_list, start=1):
            insertion = curr_row[j - 1] + 1
            deletion = prev_row[j] + 1
            substitution = prev_row[j - 1] + (0 if ref_item == hyp_item else 1)
            curr_row.append(min(insertion, deletion, substitution))
        prev_row = curr_row
    return prev_row[-1]


def compute_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    ref_words = normalize_words(reference)
    hyp_words = normalize_words(hypothesis)
    word_distance = levenshtein(ref_words, hyp_words)
    wer = word_distance / len(ref_words) if ref_words else math.nan

    ref_chars = normalize_chars(reference)
    hyp_chars = normalize_chars(hypothesis)
    char_distance = levenshtein(ref_chars, hyp_chars)

    return {
        "wer": wer,
        "char_diff": char_distance,
        "ref_word_count": len(ref_words),
        "ref_char_count": len(ref_chars),
    }


def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%" if not math.isnan(value) else "n/d"


def run_model_on_samples(
    model_name: str,
    samples: Dict[str, Sample],
    device: str,
    compute_type: str,
) -> Dict[str, dict]:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:  # pragma: no cover - fallback when package missing
        raise RuntimeError(
            "Pakiet faster-whisper nie jest dostępny. Zainstaluj zależności lub uruchom bez --run-models"
        ) from exc

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    results: Dict[str, dict] = {}

    if torch is not None and torch.cuda.is_available():  # pragma: no branch - zależy od środowiska
        torch.cuda.reset_peak_memory_stats()

    for sample in samples.values():
        if sample.audio_path is None:
            raise ValueError(
                f"Próbka {sample.sample_id} nie ma przypisanej ścieżki audio. Dodaj nagranie przed --run-models"
            )
        start = time.perf_counter()
        segments, _ = model.transcribe(str(sample.audio_path))
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        runtime = time.perf_counter() - start
        max_vram = None
        if torch is not None and torch.cuda.is_available():  # pragma: no branch - zależy od środowiska
            max_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats()
        results[sample.sample_id] = {
            "sample_id": sample.sample_id,
            "transcript": transcript,
            "runtime_s": runtime,
            "max_vram_mib": max_vram,
        }
    return results


def main() -> None:
    args = parse_args()
    samples = load_manifest(args.manifest)

    if args.run_models:
        print("Uruchamiam modele faster-whisper na lokalnych próbkach...")
        models_data: Dict[str, Dict[str, dict]] = {}
        for model_name in args.models:
            print(f"- model {model_name}")
            runs = run_model_on_samples(model_name, samples, args.device, args.compute_type)
            models_data[model_name] = {
                "device": args.device,
                "compute_type": args.compute_type,
                "runs": runs,
            }
    else:
        models_data = load_precomputed(args.precomputed)

    report = {"models": {}, "samples": {sample_id: sample.category for sample_id, sample in samples.items()}}
    table_rows: List[str] = []
    header = (
        "Model",
        "Próbka",
        "WER",
        "Char diff",
        "Czas [s]",
        "VRAM [MiB]",
    )
    table_rows.append(" | ".join(header))
    table_rows.append(" | ".join("---" for _ in header))

    for model_name, data in models_data.items():
        runs = data["runs"]
        metrics_per_sample = {}
        wer_values = []
        char_values = []
        runtime_values = []
        vram_values = []
        for sample_id, sample in samples.items():
            run = runs.get(sample_id)
            if not run:
                raise KeyError(f"Brak wyników modelu {model_name} dla próbki {sample_id}")
            metrics = compute_metrics(sample.reference_text, run["transcript"])
            metrics_per_sample[sample_id] = {
                **metrics,
                "runtime_s": run.get("runtime_s"),
                "max_vram_mib": run.get("max_vram_mib"),
            }
            if not math.isnan(metrics["wer"]):
                wer_values.append(metrics["wer"])
            char_values.append(metrics["char_diff"])
            if run.get("runtime_s") is not None:
                runtime_values.append(run["runtime_s"])
            if run.get("max_vram_mib") is not None:
                vram_values.append(run["max_vram_mib"])
            table_rows.append(
                " | ".join(
                    [
                        model_name,
                        sample_id,
                        format_percentage(metrics["wer"]),
                        str(metrics["char_diff"]),
                        f"{run.get('runtime_s', 'n/d'):.1f}" if run.get("runtime_s") else "n/d",
                        f"{run.get('max_vram_mib', 'n/d'):.0f}" if run.get("max_vram_mib") else "n/d",
                    ]
                )
            )
        report["models"][model_name] = {
            "device": data.get("device"),
            "compute_type": data.get("compute_type"),
            "per_sample": metrics_per_sample,
            "avg_wer": statistics.mean(wer_values) if wer_values else math.nan,
            "avg_char_diff": statistics.mean(char_values) if char_values else math.nan,
            "avg_runtime_s": statistics.mean(runtime_values) if runtime_values else None,
            "max_vram_mib": max(vram_values) if vram_values else None,
        }

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nPodsumowanie:")
    print("\n".join(table_rows))
    print(f"\nZapisano metryki do {args.output}")


if __name__ == "__main__":
    main()
