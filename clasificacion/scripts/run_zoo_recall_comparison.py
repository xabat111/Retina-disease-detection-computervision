"""Ejecuta comparación + Grad-CAM (+ ensemble opcional) del notebook 07 sin reentrenar.

Requiere los 4 checkpoints recallfocus_*_nocrop.pt. No sobrescribe checkpoints.
Uso:  python -u scripts/run_zoo_recall_comparison.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "artifacts" / "torch_cache"))
sys.stdout.reconfigure(encoding="utf-8")

# Headless: evita bloquear en plt.show()
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.cached_datasets import build_cached_letterbox_splits  # noqa: E402
from src.comparison_plots import plot_metric_bars  # noqa: E402
from src.data import PREPROCESSING_LETTERBOX  # noqa: E402
from src.ensemble_evaluation import (  # noqa: E402
    FN_BELOW_FP_RULE,
    RECALL_TARGET_RULE,
    average_over_members,
    average_over_views,
    build_model_comparison_table,
    choose_operating_thresholds,
    load_member_views,
    summary_for_json,
)
from src.evaluate import select_threshold_youden  # noqa: E402
from src.gradcam_compare import GROUP_ORDER, compute_gradcam_for_selection, plot_gradcam_model_comparison  # noqa: E402
from src.hard_example_plots import plot_training_histories  # noqa: E402
from src.models_zoo import get_architecture_spec, load_architecture_checkpoint  # noqa: E402
from src.recall_config import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    PREDICTION_CACHE_DIR as PREVIOUS_PREDICTION_CACHE_DIR,
    TARGET_RECALL,
    VALIDATION_RECALL_MARGIN,
    existing_checkpoint_members,
)
from src.recall_focus_config import (  # noqa: E402
    CHECKPOINT_DIR as RESNET152_CHECKPOINT_DIR,
    PREDICTION_CACHE_DIR as RECALL_FOCUS_CACHE_DIR,
    SELECTED_GRADCAM_IMAGES_PATH,
    SUMMARY_PATH as RECALL_FOCUS_SUMMARY_PATH,
    VARIANTS,
    WARM_START_CHECKPOINT,
)
from src.recall_optimization import (  # noqa: E402
    bootstrap_metric_differences,
    choose_ensemble_subset,
    compute_operating_point_metrics,
    rank_ensemble_subsets,
)
from src.recall_plots import plot_confusion_matrices, plot_roc_and_pr_curves  # noqa: E402
from src.recall_report import build_summary_table, summarize_operating_point  # noqa: E402
from src.train import get_device  # noqa: E402
from src.tta_inference import DEFAULT_TTA_VIEW_NAMES, EnsembleMember  # noqa: E402
from src.zoo_recall_config import (  # noqa: E402
    ARCHITECTURES,
    CHECKPOINT_DIR,
    FIGURES_DIR,
    GRADCAM_DIR,
    PREDICTION_CACHE_DIR,
    SUMMARY_PATH,
    ZOO_EPOCHS,
    ZOO_SETTINGS,
    build_zoo_train_config,
    chosen_recall_recipe_name,
    history_path,
    recall_recipe,
)


def _require_zoo_checkpoints() -> None:
    missing = []
    for architecture in ARCHITECTURES:
        checkpoint = CHECKPOINT_DIR / ZOO_SETTINGS[architecture].checkpoint_name(architecture)
        history = history_path(architecture)
        if not checkpoint.is_file() or not history.is_file():
            missing.append(architecture)
    if missing:
        raise FileNotFoundError(
            "Faltan checkpoint+historial para: "
            + ", ".join(missing)
            + ". Espera a que termine scripts/train_zoo_recall.py."
        )


def main() -> None:
    _require_zoo_checkpoints()

    device = get_device()
    validation_recall_goal = TARGET_RECALL + VALIDATION_RECALL_MARGIN
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={device}", flush=True)

    recipe_alpha, _ = recall_recipe()
    splits = build_cached_letterbox_splits(build_zoo_train_config("resnet50"))
    val_labels = splits.val_dataframe["disease_risk"].to_numpy(dtype=float)
    test_labels = splits.test_dataframe["disease_risk"].to_numpy(dtype=float)
    labels_by_split = {"val": val_labels, "test": test_labels}
    dataframes = {"val": splits.val_dataframe, "test": splits.test_dataframe}

    histories = {
        architecture: json.loads(history_path(architecture).read_text(encoding="utf-8"))
        for architecture in ARCHITECTURES
    }
    time_table = pd.DataFrame(
        [
            {
                "modelo": get_architecture_spec(architecture).display_name,
                "épocas": len(history),
                "s/época (train+val)": float(np.mean([row["epoch_seconds"] for row in history])),
                "horas totales": sum(row["epoch_seconds"] for row in history) / 3600,
                "mejor época (val)": max(history, key=lambda row: row["val_spec_at_fn_lt_fp"])["epoch"],
                "spec@FN<FP val": max(history, key=lambda row: row["val_spec_at_fn_lt_fp"])["val_spec_at_fn_lt_fp"],
                "AUC val (mejor)": max(history, key=lambda row: row["val_spec_at_fn_lt_fp"])["val_auc"],
            }
            for architecture, history in histories.items()
        ]
    ).set_index("modelo")
    print(time_table.round(4).to_string())
    print(f"Tiempo total de entrenamiento: {time_table['horas totales'].sum():.2f} h", flush=True)
    plot_training_histories(
        {get_architecture_spec(architecture).display_name: history for architecture, history in histories.items()},
        save_path=FIGURES_DIR / "training_curves.png",
    )
    plt.close("all")

    reference_member = EnsembleMember(
        "resnet152_optuna_nocrop",
        WARM_START_CHECKPOINT,
        PREPROCESSING_LETTERBOX,
        architecture="resnet152",
    )
    zoo_members = [
        EnsembleMember(
            f"zoo_{architecture}",
            CHECKPOINT_DIR / ZOO_SETTINGS[architecture].checkpoint_name(architecture),
            PREPROCESSING_LETTERBOX,
            architecture=architecture,
        )
        for architecture in ARCHITECTURES
    ]
    comparison_members = [reference_member, *zoo_members]
    missing = [member.name for member in comparison_members if not member.checkpoint_path.is_file()]
    if missing:
        raise FileNotFoundError("Faltan checkpoints para la comparación: " + ", ".join(missing))

    print("Cargando predicciones (identity)...", flush=True)
    plain_views = load_member_views(
        comparison_members, dataframes, PREDICTION_CACHE_DIR, device, view_names=("identity",)
    )
    plain_probabilities = average_over_views(plain_views, use_tta=False)

    display_names = {
        "resnet152_optuna_nocrop": "ResNet-152 optuna nocrop (ref.)",
        **{f"zoo_{architecture}": get_architecture_spec(architecture).display_name for architecture in ARCHITECTURES},
    }

    comparison_rows = []
    for member in comparison_members:
        name = member.name
        threshold = select_threshold_youden(plain_probabilities[name]["val"], val_labels)
        val_metrics = compute_operating_point_metrics(plain_probabilities[name]["val"], val_labels, threshold)
        test_metrics = compute_operating_point_metrics(plain_probabilities[name]["test"], test_labels, threshold)
        (test_tn, test_fp), (test_fn, test_tp) = test_metrics["confusion_matrix"]
        (_, val_fp), (val_fn, _) = val_metrics["confusion_matrix"]
        comparison_rows.append(
            {
                "modelo": display_names[name],
                "umbral (val, Youden)": threshold,
                "AUC val": val_metrics["auc"],
                "AUC test": test_metrics["auc"],
                "F1 test": test_metrics["f1"],
                "Accuracy test": test_metrics["accuracy"],
                "Precision test": test_metrics["precision"],
                "Recall test": test_metrics["recall"],
                "Especificidad test": test_metrics["specificity"],
                "val FN / FP": f"{val_fn} / {val_fp}",
                "TN": test_tn,
                "FP": test_fp,
                "FN": test_fn,
                "TP": test_tp,
            }
        )

    comparison_table = pd.DataFrame(comparison_rows).sort_values("AUC test", ascending=False).reset_index(drop=True)
    comparison_table.to_csv(FIGURES_DIR / "compare_metrics.csv", index=False)
    print("\n=== Métricas (Youden, sin TTA) ===", flush=True)
    print(comparison_table.round(4).to_string(index=False), flush=True)

    plot_metric_bars(comparison_table, save_path=FIGURES_DIR / "compare_metric_bars.png")
    plt.close("all")
    plot_roc_and_pr_curves(
        curves={
            display_names[name]: (plain_probabilities[name]["test"], test_labels)
            for name in display_names
        },
        operating_points={},
        title="ROC y Precisión-Recall en test (sin TTA; umbral Youden de validación)",
        save_path=FIGURES_DIR / "compare_roc_pr_test.png",
    )
    plt.close("all")
    plot_confusion_matrices(
        {
            display_names[name]: compute_operating_point_metrics(
                plain_probabilities[name]["test"],
                test_labels,
                select_threshold_youden(plain_probabilities[name]["val"], val_labels),
            )["confusion_matrix"]
            for name in display_names
        },
        title="Matrices de confusión en test (umbral Youden de validación)",
        save_path=FIGURES_DIR / "compare_confusion_matrices_test.png",
    )
    plt.close("all")
    print("Figuras comparación escritas en", FIGURES_DIR, flush=True)

    selected_images = json.loads(SELECTED_GRADCAM_IMAGES_PATH.read_text(encoding="utf-8"))["groups"]
    gradcam_by_model = {}
    decision_thresholds = {}
    for member in comparison_members:
        display_name = display_names[member.name]
        print(f"Grad-CAM: {display_name}", flush=True)
        model = load_architecture_checkpoint(member.architecture, member.checkpoint_path, device)
        gradcam_by_model[display_name] = compute_gradcam_for_selection(
            model, member.architecture, selected_images, member.image_size, PREPROCESSING_LETTERBOX
        )
        decision_thresholds[display_name] = select_threshold_youden(
            plain_probabilities[member.name]["val"], val_labels
        )
        del model

    for group_name in GROUP_ORDER:
        plot_gradcam_model_comparison(
            group_name,
            selected_images,
            gradcam_by_model,
            decision_thresholds,
            save_path=GRADCAM_DIR / f"compare_{group_name}.png",
        )
        plt.close("all")
    print("Grad-CAM grids en", GRADCAM_DIR, flush=True)

    # --- Sección opcional: ensemble + TTA ---
    configurations = {}
    test_summaries = {}
    val_summaries = {}
    validation_best_subset = ()
    paired = {}
    try:
        recall_focus_summary = json.loads(RECALL_FOCUS_SUMMARY_PATH.read_text(encoding="utf-8"))
        resnet152_recall_name = f"recallfocus_{recall_focus_summary['chosen_variant_validation']}"
        resnet152_recall_member = EnsembleMember(
            resnet152_recall_name,
            RESNET152_CHECKPOINT_DIR / VARIANTS[recall_focus_summary["chosen_variant_validation"]].checkpoint_name,
            PREPROCESSING_LETTERBOX,
        )
        previous_members = existing_checkpoint_members()
        previous_member_names = tuple(member.name for member in previous_members)

        print("Cargando vistas TTA (reutiliza .npz)...", flush=True)
        all_views = {
            **load_member_views(previous_members, dataframes, PREVIOUS_PREDICTION_CACHE_DIR, device),
            **load_member_views([resnet152_recall_member], dataframes, RECALL_FOCUS_CACHE_DIR, device),
            **load_member_views(zoo_members, dataframes, PREDICTION_CACHE_DIR, device),
        }
        tta_probabilities = average_over_views(all_views, use_tta=True)

        tta_display_names = {
            "ajustado_nocrop": "ResNet-152 ajustado (05)",
            resnet152_recall_name: (
                f"ResNet-152 recall-focus (06: {recall_focus_summary['chosen_variant_validation']})"
            ),
            **{
                f"zoo_{architecture}": get_architecture_spec(architecture).display_name + " (07)"
                for architecture in ARCHITECTURES
            },
        }

        individual_tta_table = build_model_comparison_table(
            {tta_display_names[name]: tta_probabilities[name] for name in tta_display_names},
            labels_by_split,
            FN_BELOW_FP_RULE,
            validation_recall_goal,
            BOOTSTRAP_SEED,
        )
        print("\n=== Individuales con TTA (FN < FP) ===", flush=True)
        print(individual_tta_table.round(4).to_string(index=False), flush=True)
        plot_roc_and_pr_curves(
            curves={
                tta_display_names[name]: (tta_probabilities[name]["test"], test_labels)
                for name in tta_display_names
            },
            operating_points={},
            title="ROC / PR en test con TTA (sección opcional)",
            save_path=FIGURES_DIR / "roc_pr_test_tta_optional.png",
        )
        plt.close("all")

        zoo_member_names = tuple(member.name for member in zoo_members)
        ranking = rank_ensemble_subsets(
            {name: probabilities["val"] for name, probabilities in tta_probabilities.items()},
            val_labels,
        )
        validation_best_subset = choose_ensemble_subset(ranking)
        print(f"Subconjunto elegido con VALIDACIÓN: {validation_best_subset}", flush=True)

        best_of_06_members = tuple(recall_focus_summary["ensemble_members"])
        systems = {
            "Anterior (05): 4 ResNet-152": average_over_members(tta_probabilities, previous_member_names),
            "Mejor de 06": average_over_members(tta_probabilities, best_of_06_members),
            "Zoo: 4 modelos nuevos": average_over_members(tta_probabilities, zoo_member_names),
            "4 nuevos + ResNet-152 recall-focus": average_over_members(
                tta_probabilities, zoo_member_names + (resnet152_recall_name,)
            ),
            "Subconjunto elegido por AUC val": average_over_members(tta_probabilities, validation_best_subset),
        }

        for system_name, probabilities in systems.items():
            thresholds = choose_operating_thresholds(
                probabilities["val"], val_labels, validation_recall_goal, BOOTSTRAP_SEED
            )
            for rule_name, threshold in thresholds.items():
                configurations[f"{system_name} | {rule_name}"] = (probabilities, threshold)

        for label, (probabilities, threshold) in configurations.items():
            val_summaries[label] = summarize_operating_point(
                probabilities["val"], val_labels, threshold, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, with_intervals=False
            )
            test_summaries[label] = summarize_operating_point(
                probabilities["test"], test_labels, threshold, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
            )

        for rule_name in (FN_BELOW_FP_RULE, RECALL_TARGET_RULE):
            print(f"\nTEST ensemble (IC 95%), regla: {rule_name}", flush=True)
            print(
                build_summary_table(
                    {key: value for key, value in test_summaries.items() if key.endswith(rule_name)}
                ).to_string(),
                flush=True,
            )

        new_label = f"4 nuevos + ResNet-152 recall-focus | {FN_BELOW_FP_RULE}"
        best_of_06_label = f"Mejor de 06 | {FN_BELOW_FP_RULE}"
        paired = bootstrap_metric_differences(
            configurations[new_label][0]["test"],
            configurations[new_label][1],
            configurations[best_of_06_label][0]["test"],
            configurations[best_of_06_label][1],
            test_labels,
            n_resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED,
        )
        print("Diferencias pareadas (4 nuevos+RN152 rf − Mejor de 06), regla FN < FP", flush=True)
        print(
            pd.DataFrame(
                {
                    metric: {"diferencia": difference, "IC95 inf": lo, "IC95 sup": hi}
                    for metric, (difference, lo, hi) in paired.items()
                }
            )
            .T.round(4)
            .to_string(),
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 — sección opcional no debe tumbar la comparación
        print(f"[aviso] Sección ensemble+TTA omitida o falló: {exc}", flush=True)
        configurations = {}
        test_summaries = {}
        paired = {}

    individual_test_rows = {}
    for member in comparison_members:
        name = member.name
        threshold = select_threshold_youden(plain_probabilities[name]["val"], val_labels)
        individual_test_rows[display_names[name]] = summary_for_json(
            summarize_operating_point(
                plain_probabilities[name]["test"], test_labels, threshold, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
            )
        ) | {"threshold_rule": "Youden (validation)", "threshold": threshold}

    optional_ensemble = {}
    if configurations and test_summaries:
        optional_ensemble = {
            "ensemble_members_validation_best": list(validation_best_subset),
            "thresholds_optional_ensemble": {label: configurations[label][1] for label in configurations},
            "val_metrics_optional_ensemble": {label: summary_for_json(s) for label, s in val_summaries.items()},
            "test_metrics_optional_ensemble": {label: summary_for_json(s) for label, s in test_summaries.items()},
            "paired_difference_optional": {
                metric: {"difference": difference, "ci95": [lo, hi]}
                for metric, (difference, lo, hi) in paired.items()
            },
        }

    summary = {
        "goal": "entrenar zoo sin Optuna (letterbox + recall) y comparar como el notebook 04; ensemble+TTA opcional",
        "recipe": {
            "variant_from_06": chosen_recall_recipe_name(),
            "focal_alpha": recipe_alpha,
            "epochs": ZOO_EPOCHS,
        },
        "hyperparameters": {
            architecture: {
                "learning_rate": settings.learning_rate,
                "weight_decay": settings.weight_decay,
                "unfrozen_blocks": settings.unfrozen_blocks,
                "rationale": settings.rationale,
            }
            for architecture, settings in ZOO_SETTINGS.items()
        },
        "comparison_table_youden_no_tta": comparison_table.to_dict(orient="records"),
        "individual_models_youden_test": individual_test_rows,
        "tta_views": list(DEFAULT_TTA_VIEW_NAMES),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "confidence_level": 0.95, "seed": BOOTSTRAP_SEED},
        **optional_ensemble,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Guardado en", SUMMARY_PATH, flush=True)


if __name__ == "__main__":
    main()
