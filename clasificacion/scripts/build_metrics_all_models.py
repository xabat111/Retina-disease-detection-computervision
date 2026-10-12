"""Consolida métricas de todos los modelos desde JSON autoritativos y salidas de nb07."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FN_BELOW_FP_RULE = "FN < FP (90 % bootstrap val)"
RECALL_TARGET_RULE = "recall val >= 0.91"
YOUDEN_RULE = "Youden (validation)"


def _cm_to_counts(confusion: list[int]) -> dict[str, int]:
    tn, fp, fn, tp = confusion
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


def _row_from_metrics(
    *,
    section: str,
    model: str,
    split: str,
    threshold_rule: str,
    metrics: dict,
    source: str,
    spec_at_fn_lt_fp_val: float | None = None,
    val_fn_fp: str | None = None,
) -> dict:
    cm = metrics.get("confusion_matrix_TN_FP_FN_TP")
    counts = _cm_to_counts(cm) if cm else {}
    return {
        "section": section,
        "model_or_config": model,
        "split": split,
        "threshold_rule": threshold_rule,
        "threshold": metrics.get("threshold"),
        "AUC": metrics.get("auc"),
        "recall": metrics.get("recall"),
        "specificity": metrics.get("specificity"),
        "precision": metrics.get("precision"),
        "accuracy": metrics.get("accuracy"),
        "F1": metrics.get("f1"),
        "TN": counts.get("TN"),
        "FP": counts.get("FP"),
        "FN": counts.get("FN"),
        "TP": counts.get("TP"),
        "val_FN_FP": val_fn_fp,
        "spec_at_FN_lt_FP_val": spec_at_fn_lt_fp_val,
        "source": source,
    }


def _split_config_label(label: str) -> tuple[str, str]:
    if " | " in label:
        config, rule = label.rsplit(" | ", 1)
        return config.strip(), rule.strip()
    return label.strip(), label.strip()


def _rows_from_labeled_metrics(
    section: str,
    split: str,
    labeled: dict[str, dict],
    source: str,
    default_rule: str | None = None,
) -> list[dict]:
    rows = []
    for label, metrics in labeled.items():
        model, parsed_rule = _split_config_label(label)
        rule = metrics.get("threshold_rule") or default_rule or parsed_rule
        rows.append(
            _row_from_metrics(
                section=section,
                model=model,
                split=split,
                threshold_rule=rule,
                metrics=metrics,
                source=source,
            )
        )
    return rows


def _metrics_from_confusion(threshold: float, tn: int, fp: int, fn: int, tp: int) -> dict:
    total = tn + fp + fn + tp
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    accuracy = (tn + tp) / total if total else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else float("nan")
    return {
        "threshold": threshold,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "accuracy": accuracy,
        "f1": f1,
        "confusion_matrix_TN_FP_FN_TP": [tn, fp, fn, tp],
    }


def _load_json(relative: str) -> dict:
    path = PROJECT_ROOT / relative
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rows() -> list[dict]:
    rows: list[dict] = []

    recall_05 = _load_json("artifacts/recall/final_summary.json")
    rows.extend(
        _rows_from_labeled_metrics(
            "05_recall_stages",
            "val",
            recall_05["val_metrics"],
            "artifacts/recall/final_summary.json",
        )
    )
    rows.extend(
        _rows_from_labeled_metrics(
            "05_recall_stages",
            "test",
            recall_05["test_metrics"],
            "artifacts/recall/final_summary.json",
        )
    )

    recall_focus = _load_json("artifacts/recall_focus/summary.json")
    for variant_name, variant in recall_focus["variants"].items():
        rows.append(
            {
                "section": "06_recall_focus_training",
                "model_or_config": f"ResNet-152 recall-focus ({variant_name})",
                "split": "val",
                "threshold_rule": "checkpoint selection (spec@FN<FP)",
                "threshold": None,
                "AUC": variant.get("val_auc"),
                "recall": None,
                "specificity": None,
                "precision": None,
                "accuracy": None,
                "F1": None,
                "TN": None,
                "FP": None,
                "FN": None,
                "TP": None,
                "val_FN_FP": None,
                "spec_at_FN_lt_FP_val": variant.get("val_spec_at_fn_lt_fp"),
                "source": "artifacts/recall_focus/summary.json",
            }
        )

    rows.extend(
        _rows_from_labeled_metrics(
            "06_ensemble_tta",
            "val",
            recall_focus["val_metrics"],
            "artifacts/recall_focus/summary.json",
        )
    )
    rows.extend(
        _rows_from_labeled_metrics(
            "06_ensemble_tta",
            "test",
            recall_focus["test_metrics"],
            "artifacts/recall_focus/summary.json",
        )
    )

    zoo = _load_json("artifacts/model_zoo_recall/summary.json")
    for record in zoo["comparison_table_youden_no_tta"]:
        model = record["modelo"]
        threshold = record["umbral (val, Youden)"]
        test_metrics = {
            "threshold": threshold,
            "auc": record["AUC test"],
            "recall": record["Recall test"],
            "specificity": record["Especificidad test"],
            "precision": record["Precision test"],
            "accuracy": record["Accuracy test"],
            "f1": record["F1 test"],
            "confusion_matrix_TN_FP_FN_TP": [record["TN"], record["FP"], record["FN"], record["TP"]],
            "threshold_rule": YOUDEN_RULE,
        }
        rows.append(
            _row_from_metrics(
                section="07_individual_no_tta",
                model=model,
                split="test",
                threshold_rule=YOUDEN_RULE,
                metrics=test_metrics,
                source="artifacts/model_zoo_recall/summary.json",
                val_fn_fp=record.get("val FN / FP"),
            )
        )
        rows.append(
            {
                "section": "07_individual_no_tta",
                "model_or_config": model,
                "split": "val",
                "threshold_rule": YOUDEN_RULE,
                "threshold": threshold,
                "AUC": record["AUC val"],
                "recall": None,
                "specificity": None,
                "precision": None,
                "accuracy": None,
                "F1": None,
                "TN": None,
                "FP": None,
                "FN": None,
                "TP": None,
                "val_FN_FP": record.get("val FN / FP"),
                "spec_at_FN_lt_FP_val": None,
                "source": "artifacts/model_zoo_recall/summary.json",
            }
        )

    zoo_training = [
        ("ResNet-50", 0.6950, 0.9243),
        ("DenseNet-121", 0.7248, 0.9392),
        ("EfficientNet-B0", 0.7099, 0.9388),
        ("ConvNeXt-Tiny", 0.7472, 0.9461),
    ]
    for model, spec_fn, auc_val in zoo_training:
        rows.append(
            {
                "section": "07_zoo_training_checkpoint",
                "model_or_config": model,
                "split": "val",
                "threshold_rule": "checkpoint selection (spec@FN<FP)",
                "threshold": None,
                "AUC": auc_val,
                "recall": None,
                "specificity": None,
                "precision": None,
                "accuracy": None,
                "F1": None,
                "TN": None,
                "FP": None,
                "FN": None,
                "TP": None,
                "val_FN_FP": None,
                "spec_at_FN_lt_FP_val": spec_fn,
                "source": "artifacts/logs/nb07_exec.out",
            }
        )

    tta_fn_fp_nb07 = [
        ("ResNet-152 ajustado (05)", 0.310, 108, 26, 27, 479),
        ("ResNet-152 recall-focus (06: control)", 0.335, 95, 39, 21, 485),
        ("ResNet-50 (07)", 0.515, 100, 34, 29, 477),
        ("DenseNet-121 (07)", 0.380, 108, 26, 25, 481),
        ("EfficientNet-B0 (07)", 0.463, 106, 28, 31, 475),
        ("ConvNeXt-Tiny (07)", 0.482, 110, 24, 27, 479),
    ]
    for model, threshold, tn, fp, fn, tp in tta_fn_fp_nb07:
        metrics = _metrics_from_confusion(threshold, tn, fp, fn, tp)
        metrics["threshold_rule"] = FN_BELOW_FP_RULE
        rows.append(
            _row_from_metrics(
                section="07_individual_tta",
                model=model,
                split="test",
                threshold_rule=FN_BELOW_FP_RULE,
                metrics=metrics,
                source="artifacts/logs/nb07_exec.out (TTA; AUC no exportado en consola)",
            )
        )

    rows.extend(
        _rows_from_labeled_metrics(
            "07_ensemble_tta",
            "val",
            zoo["val_metrics_optional_ensemble"],
            "artifacts/model_zoo_recall/summary.json",
        )
    )
    rows.extend(
        _rows_from_labeled_metrics(
            "07_ensemble_tta",
            "test",
            zoo["test_metrics_optional_ensemble"],
            "artifacts/model_zoo_recall/summary.json",
        )
    )

    return rows


def _format_cell(value) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.4f}"
    return str(value)


def _format_md_table(df: pd.DataFrame, columns: list[str]) -> str:
    display = df[columns].copy()
    for col in display.columns:
        display[col] = display[col].map(_format_cell)
    headers = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in display.to_numpy())
    return "\n".join([headers, sep, body])


def write_outputs(rows: list[dict]) -> tuple[Path, Path]:
    df = pd.DataFrame(rows)
    column_order = [
        "section",
        "model_or_config",
        "split",
        "threshold_rule",
        "threshold",
        "AUC",
        "recall",
        "specificity",
        "precision",
        "accuracy",
        "F1",
        "TN",
        "FP",
        "FN",
        "TP",
        "val_FN_FP",
        "spec_at_FN_lt_FP_val",
        "source",
    ]
    df = df[column_order]

    csv_path = PROJECT_ROOT / "results" / "current" / "metrics_all_models.csv"
    md_path = PROJECT_ROOT / "docs" / "METRICAS_MODELOS.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format="%.10g")

    md_cols = [
        "model_or_config",
        "split",
        "threshold_rule",
        "threshold",
        "AUC",
        "recall",
        "specificity",
        "precision",
        "accuracy",
        "F1",
        "TN",
        "FP",
        "FN",
        "TP",
    ]

    sections = [
        ("07_individual_no_tta", "Modelos individuales sin TTA (Youden en validación)"),
        ("07_individual_tta", "Modelos individuales con TTA (FN < FP, umbral en validación)"),
        ("07_ensemble_tta", "Ensembles con TTA (notebooks 05 / 06 / 07)"),
        ("06_ensemble_tta", "Ensemble recall-focus (notebook 06)"),
        ("05_recall_stages", "Etapas del pipeline de recall (notebook 05, referencia ResNet-152)"),
        ("06_recall_focus_training", "Entrenamiento recall-focus ResNet-152 (selección de checkpoint)"),
        ("07_zoo_training_checkpoint", "Zoo (07): criterio de checkpoint en validación"),
    ]

    lines = [
        "# Métricas de todos los modelos y configuraciones",
        "",
        "Tabla consolidada desde `artifacts/recall/final_summary.json`, "
        "`artifacts/recall_focus/summary.json`, `artifacts/model_zoo_recall/summary.json`, "
        "`results/experiments/model_zoo_recall/figures/compare_metrics.csv` y salida de nb07.",
        "",
        "Criterios del proyecto: **FN < FP** (bootstrap 90 % en val) y **recall val ≥ 0.91**; "
        "comparaciones individuales del zoo usan **Youden en validación**.",
        "",
        f"CSV completo: `{csv_path.relative_to(PROJECT_ROOT).as_posix()}` ({len(df)} filas).",
        "",
    ]

    for section_key, title in sections:
        part = df[df["section"] == section_key]
        if part.empty:
            continue
        lines.append(f"## {title}")
        lines.append("")
        if section_key == "07_individual_tta":
            lines.append(
                "> AUC no aparece en la salida de consola de nb07; recall/especificidad/F1 se derivan "
                "de TN/FP/FN/TP documentados en `artifacts/logs/nb07_exec.out`."
            )
            lines.append("")
        lines.append(_format_md_table(part, md_cols))
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    rows = collect_rows()
    csv_path, md_path = write_outputs(rows)
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
