# Métricas de todos los modelos y configuraciones

Tabla consolidada desde `artifacts/recall/final_summary.json`, `artifacts/recall_focus/summary.json`, `artifacts/model_zoo_recall/summary.json`, `results/experiments/model_zoo_recall/figures/compare_metrics.csv` y salida de nb07.

Criterios del proyecto: **FN < FP** (bootstrap 90 % en val) y **recall val ≥ 0.91**; comparaciones individuales del zoo usan **Youden en validación**.

CSV completo: `results/current/metrics_all_models.csv` (62 filas).

## Modelos individuales sin TTA (Youden en validación)

| model_or_config | split | threshold_rule | threshold | AUC | recall | specificity | precision | accuracy | F1 | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-152 optuna nocrop (ref.) | test | Youden (validation) | 0.5447 | 0.9677 | 0.8498 | 0.9627 | 0.9885 | 0.8734 | 0.9139 | 129 | 5 | 76 | 430 |
| ResNet-152 optuna nocrop (ref.) | val | Youden (validation) | 0.5447 | 0.9375 | — | — | — | — | — | — | — | — | — |
| DenseNet-121 | test | Youden (validation) | 0.6661 | 0.9601 | 0.8636 | 0.8806 | 0.9647 | 0.8672 | 0.9114 | 118 | 16 | 69 | 437 |
| DenseNet-121 | val | Youden (validation) | 0.6661 | 0.9392 | — | — | — | — | — | — | — | — | — |
| ConvNeXt-Tiny | test | Youden (validation) | 0.6184 | 0.9589 | 0.7964 | 0.9776 | 0.9926 | 0.8344 | 0.8838 | 131 | 3 | 103 | 403 |
| ConvNeXt-Tiny | val | Youden (validation) | 0.6184 | 0.9461 | — | — | — | — | — | — | — | — | — |
| EfficientNet-B0 | test | Youden (validation) | 0.5507 | 0.9503 | 0.8676 | 0.9104 | 0.9734 | 0.8766 | 0.9175 | 122 | 12 | 67 | 439 |
| EfficientNet-B0 | val | Youden (validation) | 0.5507 | 0.9388 | — | — | — | — | — | — | — | — | — |
| ResNet-50 | test | Youden (validation) | 0.5717 | 0.9502 | 0.8755 | 0.9104 | 0.9736 | 0.8828 | 0.9220 | 122 | 12 | 63 | 443 |
| ResNet-50 | val | Youden (validation) | 0.5717 | 0.9243 | — | — | — | — | — | — | — | — | — |

## Modelos individuales con TTA (FN < FP, umbral en validación)

> AUC recalculado con `roc_auc_score` sobre la media TTA de las cachés `.npz` (`artifacts/logs/individual_tta_aucs.json`).

| model_or_config | split | threshold_rule | threshold | AUC | recall | specificity | precision | accuracy | F1 | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-152 ajustado (05) | test | FN < FP (90 % bootstrap val) | 0.3100 | 0.9679 | 0.9466 | 0.8060 | 0.9485 | 0.9172 | 0.9476 | 108 | 26 | 27 | 479 |
| ResNet-152 recall-focus (06: control) | test | FN < FP (90 % bootstrap val) | 0.3350 | 0.9666 | 0.9585 | 0.7090 | 0.9256 | 0.9062 | 0.9417 | 95 | 39 | 21 | 485 |
| ResNet-50 (07) | test | FN < FP (90 % bootstrap val) | 0.5150 | 0.9468 | 0.9427 | 0.7463 | 0.9335 | 0.9016 | 0.9381 | 100 | 34 | 29 | 477 |
| DenseNet-121 (07) | test | FN < FP (90 % bootstrap val) | 0.3800 | 0.9671 | 0.9506 | 0.8060 | 0.9487 | 0.9203 | 0.9497 | 108 | 26 | 25 | 481 |
| EfficientNet-B0 (07) | test | FN < FP (90 % bootstrap val) | 0.4630 | 0.9510 | 0.9387 | 0.7910 | 0.9443 | 0.9078 | 0.9415 | 106 | 28 | 31 | 475 |
| ConvNeXt-Tiny (07) | test | FN < FP (90 % bootstrap val) | 0.4820 | 0.9580 | 0.9466 | 0.8209 | 0.9523 | 0.9203 | 0.9495 | 110 | 24 | 27 | 479 |

## Ensembles con TTA (notebooks 05 / 06 / 07)

| model_or_config | split | threshold_rule | threshold | AUC | recall | specificity | precision | accuracy | F1 | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Anterior (05): 4 ResNet-152 | val | recall val >= 0.91 | 0.3993 | 0.9440 | 0.9111 | 0.7761 | 0.9389 | 0.8828 | 0.9248 | 104 | 30 | 45 | 461 |
| Anterior (05): 4 ResNet-152 | val | FN < FP (90 % bootstrap val) | 0.3525 | 0.9440 | 0.9368 | 0.6716 | 0.9151 | 0.8812 | 0.9258 | 90 | 44 | 32 | 474 |
| Mejor de 06 | val | recall val >= 0.91 | 0.4025 | 0.9454 | 0.9111 | 0.7612 | 0.9351 | 0.8797 | 0.9229 | 102 | 32 | 45 | 461 |
| Mejor de 06 | val | FN < FP (90 % bootstrap val) | 0.3475 | 0.9454 | 0.9427 | 0.6642 | 0.9138 | 0.8844 | 0.9280 | 89 | 45 | 29 | 477 |
| Zoo: 4 modelos nuevos | val | recall val >= 0.91 | 0.5205 | 0.9490 | 0.9111 | 0.7836 | 0.9408 | 0.8844 | 0.9257 | 105 | 29 | 45 | 461 |
| Zoo: 4 modelos nuevos | val | FN < FP (90 % bootstrap val) | 0.4625 | 0.9490 | 0.9407 | 0.6866 | 0.9189 | 0.8875 | 0.9297 | 92 | 42 | 30 | 476 |
| 4 nuevos + ResNet-152 recall-focus | val | recall val >= 0.91 | 0.5030 | 0.9505 | 0.9111 | 0.8060 | 0.9466 | 0.8891 | 0.9285 | 108 | 26 | 45 | 461 |
| 4 nuevos + ResNet-152 recall-focus | val | FN < FP (90 % bootstrap val) | 0.4475 | 0.9505 | 0.9427 | 0.6940 | 0.9208 | 0.8906 | 0.9316 | 93 | 41 | 29 | 477 |
| Subconjunto elegido por AUC val | val | recall val >= 0.91 | 0.4713 | 0.9492 | 0.9111 | 0.8209 | 0.9505 | 0.8922 | 0.9304 | 110 | 24 | 45 | 461 |
| Subconjunto elegido por AUC val | val | FN < FP (90 % bootstrap val) | 0.3950 | 0.9492 | 0.9427 | 0.6791 | 0.9173 | 0.8875 | 0.9298 | 91 | 43 | 29 | 477 |
| Anterior (05): 4 ResNet-152 | test | recall val >= 0.91 | 0.3993 | 0.9706 | 0.9289 | 0.8806 | 0.9671 | 0.9187 | 0.9476 | 118 | 16 | 36 | 470 |
| Anterior (05): 4 ResNet-152 | test | FN < FP (90 % bootstrap val) | 0.3525 | 0.9706 | 0.9387 | 0.8433 | 0.9577 | 0.9187 | 0.9481 | 113 | 21 | 31 | 475 |
| Mejor de 06 | test | recall val >= 0.91 | 0.4025 | 0.9723 | 0.9308 | 0.8881 | 0.9691 | 0.9219 | 0.9496 | 119 | 15 | 35 | 471 |
| Mejor de 06 | test | FN < FP (90 % bootstrap val) | 0.3475 | 0.9723 | 0.9447 | 0.8358 | 0.9560 | 0.9219 | 0.9503 | 112 | 22 | 28 | 478 |
| Zoo: 4 modelos nuevos | test | recall val >= 0.91 | 0.5205 | 0.9714 | 0.9249 | 0.8955 | 0.9710 | 0.9187 | 0.9474 | 120 | 14 | 38 | 468 |
| Zoo: 4 modelos nuevos | test | FN < FP (90 % bootstrap val) | 0.4625 | 0.9714 | 0.9526 | 0.8060 | 0.9488 | 0.9219 | 0.9507 | 108 | 26 | 24 | 482 |
| 4 nuevos + ResNet-152 recall-focus | test | recall val >= 0.91 | 0.5030 | 0.9729 | 0.9289 | 0.8955 | 0.9711 | 0.9219 | 0.9495 | 120 | 14 | 36 | 470 |
| 4 nuevos + ResNet-152 recall-focus | test | FN < FP (90 % bootstrap val) | 0.4475 | 0.9729 | 0.9565 | 0.8134 | 0.9509 | 0.9266 | 0.9537 | 109 | 25 | 22 | 484 |
| Subconjunto elegido por AUC val | test | recall val >= 0.91 | 0.4713 | 0.9731 | 0.9170 | 0.9104 | 0.9748 | 0.9156 | 0.9450 | 122 | 12 | 42 | 464 |
| Subconjunto elegido por AUC val | test | FN < FP (90 % bootstrap val) | 0.3950 | 0.9731 | 0.9506 | 0.8060 | 0.9487 | 0.9203 | 0.9497 | 108 | 26 | 25 | 481 |

## Ensemble recall-focus (notebook 06)

| model_or_config | split | threshold_rule | threshold | AUC | recall | specificity | precision | accuracy | F1 | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Anterior (05): 4 ResNet-152 + TTA | val | recall val >= 0.91  [= FINAL de 05] | 0.3993 | 0.9440 | 0.9111 | 0.7761 | 0.9389 | 0.8828 | 0.9248 | 104 | 30 | 45 | 461 |
| Anterior (05): 4 ResNet-152 + TTA | val | FN < FP (90 % bootstrap val)  [solo umbral] | 0.3525 | 0.9440 | 0.9368 | 0.6716 | 0.9151 | 0.8812 | 0.9258 | 90 | 44 | 32 | 474 |
| Ensemble elegido + TTA | val | recall val >= 0.91 | 0.4025 | 0.9454 | 0.9111 | 0.7612 | 0.9351 | 0.8797 | 0.9229 | 102 | 32 | 45 | 461 |
| Ensemble elegido + TTA | val | FN < FP (90 % bootstrap val)  [NUEVO] | 0.3475 | 0.9454 | 0.9427 | 0.6642 | 0.9138 | 0.8844 | 0.9280 | 89 | 45 | 29 | 477 |
| Anterior (05): 4 ResNet-152 + TTA | test | recall val >= 0.91  [= FINAL de 05] | 0.3993 | 0.9706 | 0.9289 | 0.8806 | 0.9671 | 0.9187 | 0.9476 | 118 | 16 | 36 | 470 |
| Anterior (05): 4 ResNet-152 + TTA | test | FN < FP (90 % bootstrap val)  [solo umbral] | 0.3525 | 0.9706 | 0.9387 | 0.8433 | 0.9577 | 0.9187 | 0.9481 | 113 | 21 | 31 | 475 |
| Ensemble elegido + TTA | test | recall val >= 0.91 | 0.4025 | 0.9723 | 0.9308 | 0.8881 | 0.9691 | 0.9219 | 0.9496 | 119 | 15 | 35 | 471 |
| Ensemble elegido + TTA | test | FN < FP (90 % bootstrap val)  [NUEVO] | 0.3475 | 0.9723 | 0.9447 | 0.8358 | 0.9560 | 0.9219 | 0.9503 | 112 | 22 | 28 | 478 |

## Etapas del pipeline de recall (notebook 05, referencia ResNet-152)

| model_or_config | split | threshold_rule | threshold | AUC | recall | specificity | precision | accuracy | F1 | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Referencia, umbral 0.5 | val | Referencia, umbral 0.5 | 0.5000 | 0.9375 | 0.8379 | 0.8955 | 0.9680 | 0.8500 | 0.8983 | 120 | 14 | 82 | 424 |
| Referencia, umbral Youden (val) | val | Referencia, umbral Youden (val) | 0.5447 | 0.9375 | 0.8241 | 0.9478 | 0.9835 | 0.8500 | 0.8968 | 127 | 7 | 89 | 417 |
| Etapa 1: referencia + umbral val (recall objetivo) | val | Etapa 1: referencia + umbral val (recall objetivo) | 0.3650 | 0.9375 | 0.9111 | 0.7537 | 0.9332 | 0.8781 | 0.9220 | 101 | 33 | 45 | 461 |
| FINAL: D: ensemble + TTA + umbral val | val | FINAL: D: ensemble + TTA + umbral val | 0.3993 | 0.9440 | 0.9111 | 0.7761 | 0.9389 | 0.8828 | 0.9248 | 104 | 30 | 45 | 461 |
| Variante sin margen (recall val >= 0.90) | val | Variante sin margen (recall val >= 0.90) | 0.4162 | 0.9440 | 0.9012 | 0.8134 | 0.9480 | 0.8828 | 0.9240 | 109 | 25 | 50 | 456 |
| Referencia, umbral 0.5 | test | Referencia, umbral 0.5 | 0.5000 | 0.9677 | 0.8676 | 0.9552 | 0.9865 | 0.8859 | 0.9232 | 128 | 6 | 67 | 439 |
| Referencia, umbral Youden (val) | test | Referencia, umbral Youden (val) | 0.5447 | 0.9677 | 0.8498 | 0.9627 | 0.9885 | 0.8734 | 0.9139 | 129 | 5 | 76 | 430 |
| Etapa 1: referencia + umbral val (recall objetivo) | test | Etapa 1: referencia + umbral val (recall objetivo) | 0.3650 | 0.9677 | 0.9229 | 0.8657 | 0.9629 | 0.9109 | 0.9425 | 116 | 18 | 39 | 467 |
| FINAL: D: ensemble + TTA + umbral val | test | FINAL: D: ensemble + TTA + umbral val | 0.3993 | 0.9706 | 0.9289 | 0.8806 | 0.9671 | 0.9187 | 0.9476 | 118 | 16 | 36 | 470 |
| Variante sin margen (recall val >= 0.90) | test | Variante sin margen (recall val >= 0.90) | 0.4162 | 0.9706 | 0.9130 | 0.8881 | 0.9686 | 0.9078 | 0.9400 | 119 | 15 | 44 | 462 |

## Entrenamiento recall-focus ResNet-152 (selección de checkpoint)

| model_or_config | split | threshold_rule | threshold | AUC | recall | specificity | precision | accuracy | F1 | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-152 recall-focus (control) | val | checkpoint selection (spec@FN<FP) | — | 0.9373 | — | — | — | — | — | — | — | — | — |
| ResNet-152 recall-focus (reweight) | val | checkpoint selection (spec@FN<FP) | — | 0.9373 | — | — | — | — | — | — | — | — | — |
| ResNet-152 recall-focus (recallloss) | val | checkpoint selection (spec@FN<FP) | — | 0.9353 | — | — | — | — | — | — | — | — | — |
| ResNet-152 recall-focus (both) | val | checkpoint selection (spec@FN<FP) | — | 0.9353 | — | — | — | — | — | — | — | — | — |

## Zoo (07): criterio de checkpoint en validación

| model_or_config | split | threshold_rule | threshold | AUC | recall | specificity | precision | accuracy | F1 | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-50 | val | checkpoint selection (spec@FN<FP) | — | 0.9243 | — | — | — | — | — | — | — | — | — |
| DenseNet-121 | val | checkpoint selection (spec@FN<FP) | — | 0.9392 | — | — | — | — | — | — | — | — | — |
| EfficientNet-B0 | val | checkpoint selection (spec@FN<FP) | — | 0.9388 | — | — | — | — | — | — | — | — | — |
| ConvNeXt-Tiny | val | checkpoint selection (spec@FN<FP) | — | 0.9461 | — | — | — | — | — | — | — | — | — |
