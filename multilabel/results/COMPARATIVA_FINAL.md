# Comparativa final TEST (n=640) - 2026-10-30

Umbrales por etiqueta ajustados SOLO en VAL (op `fbeta2` = maximiza F2 en VAL; `recall80` = recall>=0.80 en VAL). Métricas macro sobre etiquetas (sup_k12: 13 etiquetas, sin `sano`). F2 = 5PR/(4P+R) con P/R macro (misma convención que ranking_all_models_recall). IC 95 % = bootstrap percentil, 2000 remuestras de imágenes de TEST, umbrales fijos. Ensambles = media de probabilidades, pesos iguales, 2-3 miembros de los 6 mejores individuales por F2 en VAL; selección por F2 en VAL; se excluyen ensambles con TTA/no-TTA del mismo run.

## odir7 — fbeta2

| # | Modelo | Tipo | Recall | Prec. | F1 | F2 [IC95] | AUROC [IC95] | mAP | F2 VAL |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ENS[vit_residual_attn_odir7_v2_518 + vit_residual_attn_odir7_v3fundus + vit_dinov2_odir7_v2] | ensemble3 | 0.927 | 0.610 | 0.722 | 0.840 [0.822, 0.856] | 0.954 [0.945, 0.962] | 0.844 | 0.835 |
| 2 | ENS[vit_residual_attn_odir7_v2_518 + vit_residual_attn_odir7_v3fundus + vit_residual_attn_odir7_v2_s0] | ensemble3 | 0.937 | 0.586 | 0.712 | 0.836 [0.821, 0.852] | 0.954 [0.946, 0.962] | 0.845 | 0.833 |
| 3 | ENS[vit_residual_attn_odir7_v2_518 + vit_residual_attn_odir7_v2 + vit_residual_attn_odir7_v3fundus] | ensemble3 | 0.913 | 0.622 | 0.729 | 0.835 [0.817, 0.852] | 0.954 [0.945, 0.963] | 0.849 | 0.836 |
| 4 | ENS[vit_residual_attn_odir7_v2 + vit_residual_attn_odir7_v3fundus + vit_dinov2_odir7_v2_518] | ensemble3 | 0.927 | 0.597 | 0.720 | 0.834 [0.818, 0.850] | 0.950 [0.941, 0.959] | 0.843 | 0.831 |
| 5 | ENS[vit_residual_attn_odir7_v2_518 + vit_residual_attn_odir7_v2 + vit_residual_attn_odir7_v2_s0] | ensemble3 | 0.915 | 0.611 | 0.722 | 0.832 [0.814, 0.849] | 0.951 [0.942, 0.959] | 0.831 | 0.833 |
| 6 | ENS[vit_residual_attn_odir7_v2 + vit_residual_attn_odir7_v3fundus] | ensemble2 | 0.918 | 0.603 | 0.721 | 0.831 [0.814, 0.849] | 0.948 [0.939, 0.958] | 0.844 | 0.822 |
| 7 | ENS[vit_residual_attn_odir7_v2 + vit_residual_attn_odir7_v3fundus + vit_residual_attn_odir7_v2_s0] | ensemble3 | 0.902 | 0.627 | 0.732 | 0.829 [0.811, 0.847] | 0.951 [0.943, 0.960] | 0.842 | 0.825 |
| 8 | ENS[vit_residual_attn_odir7_v2_518 + vit_residual_attn_odir7_v2 + vit_dinov2_odir7_v2] | ensemble3 | 0.921 | 0.590 | 0.709 | 0.828 [0.810, 0.845] | 0.950 [0.941, 0.959] | 0.836 | 0.829 |
| 9 | ENS[vit_residual_attn_odir7_v2_518 + vit_residual_attn_odir7_v3fundus + vit_dinov2_odir7_v2_518] | ensemble3 | 0.933 | 0.567 | 0.698 | 0.826 [0.809, 0.843] | 0.952 [0.943, 0.960] | 0.839 | 0.835 |
| 10 | ENS[vit_residual_attn_odir7_v2 + vit_residual_attn_odir7_v2_s0 + vit_dinov2_odir7_v2_518] | ensemble3 | 0.912 | 0.597 | 0.716 | 0.825 [0.806, 0.843] | 0.946 [0.937, 0.954] | 0.822 | 0.818 |
| 11 | ENS[vit_residual_attn_odir7_v2_518 + vit_residual_attn_odir7_v2] | ensemble2 | 0.902 | 0.614 | 0.724 | 0.824 [0.803, 0.844] | 0.949 [0.939, 0.958] | 0.832 | 0.831 |
| 12 | ENS[vit_residual_attn_odir7_v3fundus + vit_residual_attn_odir7_v2_s0] | ensemble2 | 0.902 | 0.608 | 0.722 | 0.823 [0.802, 0.842] | 0.947 [0.938, 0.956] | 0.833 | 0.823 |
| 13 | ENS[vit_residual_attn_odir7_v3fundus + vit_residual_attn_odir7_v2_s0 + vit_dinov2_odir7_v2] | ensemble3 | 0.909 | 0.597 | 0.715 | 0.823 [0.804, 0.841] | 0.950 [0.941, 0.958] | 0.835 | 0.819 |
| 14 | ENS[vit_residual_attn_odir7_v2_518 + vit_residual_attn_odir7_v2_s0] | ensemble2 | 0.909 | 0.596 | 0.714 | 0.823 [0.803, 0.841] | 0.948 [0.939, 0.956] | 0.822 | 0.828 |
| 15 | ENS[vit_residual_attn_odir7_v2_518 + vit_dinov2_odir7_v2_518] | ensemble2 | 0.926 | 0.566 | 0.692 | 0.822 [0.805, 0.838] | 0.942 [0.933, 0.952] | 0.815 | 0.829 |

## odir7 — recall80

| # | Modelo | Tipo | Recall | Prec. | F1 | F2 [IC95] | AUROC [IC95] | mAP | F2 VAL |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ENS[vit_residual_attn_odir7_v3fundus + vit_dinov2_odir7_v2_518] | ensemble2 | 0.837 | 0.685 | 0.740 | 0.801 [0.777, 0.823] | 0.943 [0.934, 0.953] | 0.830 | 0.775 |
| 2 | ENS[vit_residual_attn_odir7_v2 + vit_residual_attn_odir7_v3fundus + vit_dinov2_odir7_v2_518] | ensemble3 | 0.825 | 0.711 | 0.750 | 0.800 [0.775, 0.822] | 0.950 [0.941, 0.959] | 0.843 | 0.781 |
| 3 | ENS[vit_residual_attn_odir7_v3fundus + vit_residual_attn_odir7_v2_s0 + vit_dinov2_odir7_v2_518] | ensemble3 | 0.831 | 0.689 | 0.740 | 0.798 [0.775, 0.821] | 0.948 [0.939, 0.956] | 0.833 | 0.777 |
| 4 | ENS[vit_residual_attn_odir7_v2_518 + vit_residual_attn_odir7_v3fundus + vit_dinov2_odir7_v2_518] | ensemble3 | 0.823 | 0.712 | 0.752 | 0.798 [0.773, 0.820] | 0.952 [0.943, 0.960] | 0.839 | 0.780 |
| 5 | ENS[vit_residual_attn_odir7_v3fundus + vit_dinov2_odir7_v2 + vit_dinov2_odir7_v2_518] | ensemble3 | 0.826 | 0.696 | 0.739 | 0.796 [0.771, 0.819] | 0.949 [0.940, 0.957] | 0.836 | 0.777 |
| 6 | ENS[vit_residual_attn_odir7_v3fundus + vit_residual_attn_odir7_v2_s0 + vit_dinov2_odir7_v2] | ensemble3 | 0.824 | 0.700 | 0.743 | 0.796 [0.771, 0.818] | 0.950 [0.941, 0.958] | 0.835 | 0.779 |
| 7 | ENS[vit_residual_attn_odir7_v2 + vit_residual_attn_odir7_v2_s0] | ensemble2 | 0.820 | 0.700 | 0.741 | 0.793 [0.767, 0.817] | 0.946 [0.937, 0.955] | 0.821 | 0.780 |
| 8 | ENS[vit_residual_attn_odir7_v2 + vit_residual_attn_odir7_v3fundus] | ensemble2 | 0.814 | 0.713 | 0.746 | 0.791 [0.765, 0.816] | 0.948 [0.939, 0.958] | 0.844 | 0.782 |
| 9 | ENS[vit_residual_attn_odir7_v2_518 + vit_residual_attn_odir7_v2 + vit_residual_attn_odir7_v2_s0] | ensemble3 | 0.816 | 0.704 | 0.745 | 0.791 [0.766, 0.815] | 0.951 [0.942, 0.959] | 0.831 | 0.782 |
| 10 | ENS[vit_residual_attn_odir7_v2 + vit_residual_attn_odir7_v2_s0 + vit_dinov2_odir7_v2_518] | ensemble3 | 0.823 | 0.683 | 0.731 | 0.790 [0.767, 0.814] | 0.946 [0.937, 0.954] | 0.822 | 0.777 |
| 11 | ENS[vit_residual_attn_odir7_v2 + vit_residual_attn_odir7_v2_s0 + vit_dinov2_odir7_v2] | ensemble3 | 0.818 | 0.696 | 0.733 | 0.790 [0.765, 0.815] | 0.947 [0.938, 0.956] | 0.826 | 0.779 |
| 12 | ENS[vit_residual_attn_odir7_v3fundus + vit_dinov2_odir7_v2] | ensemble2 | 0.812 | 0.713 | 0.747 | 0.790 [0.763, 0.814] | 0.947 [0.938, 0.956] | 0.835 | 0.781 |
| 13 | ENS[vit_residual_attn_odir7_v2_518 + vit_residual_attn_odir7_v3fundus + vit_residual_attn_odir7_v2_s0] | ensemble3 | 0.808 | 0.723 | 0.753 | 0.790 [0.764, 0.813] | 0.954 [0.946, 0.962] | 0.845 | 0.786 |
| 14 | vit_residual_attn_odir7_v3fundus_518 | single | 0.826 | 0.669 | 0.724 | 0.789 [0.766, 0.812] | 0.933 [0.922, 0.943] | 0.815 | 0.766 |
| 15 | ENS[vit_residual_attn_odir7_v2_518 + vit_residual_attn_odir7_v2_s0 + vit_dinov2_odir7_v2_518] | ensemble3 | 0.820 | 0.683 | 0.732 | 0.789 [0.765, 0.812] | 0.946 [0.938, 0.955] | 0.819 | 0.777 |

## cluster_c — fbeta2

| # | Modelo | Tipo | Recall | Prec. | F1 | F2 [IC95] | AUROC [IC95] | mAP | F2 VAL |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v2 + vit_dinov2_cluster_c_v3fundus] | ensemble3 | 0.847 | 0.613 | 0.688 | 0.787 [0.759, 0.812] | 0.942 [0.931, 0.952] | 0.783 | 0.801 |
| 2 | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v3fundus + vit_residual_attn_cluster_c_v3fundus_518] | ensemble3 | 0.843 | 0.612 | 0.689 | 0.784 [0.757, 0.809] | 0.941 [0.929, 0.953] | 0.788 | 0.802 |
| 3 | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v2 + vit_dinov2_cluster_c_v2] | ensemble3 | 0.848 | 0.596 | 0.668 | 0.782 [0.754, 0.808] | 0.936 [0.922, 0.948] | 0.770 | 0.799 |
| 4 | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v3fundus_518 + vit_dinov2_cluster_c_v2] | ensemble3 | 0.834 | 0.626 | 0.696 | 0.782 [0.754, 0.808] | 0.941 [0.929, 0.952] | 0.788 | 0.789 |
| 5 | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v3fundus + vit_dinov2_cluster_c_v2] | ensemble3 | 0.828 | 0.637 | 0.706 | 0.781 [0.752, 0.808] | 0.944 [0.932, 0.955] | 0.792 | 0.799 |
| 6 | ENS[vit_residual_attn_cluster_c_v2 + vit_residual_attn_cluster_c_v3fundus_518 + vit_dinov2_cluster_c_v3fundus] | ensemble3 | 0.837 | 0.601 | 0.670 | 0.776 [0.748, 0.802] | 0.939 [0.928, 0.950] | 0.779 | 0.804 |
| 7 | ENS[vit_residual_attn_cluster_c_v2 + vit_residual_attn_cluster_c_v3fundus + vit_residual_attn_cluster_c_v3fundus_518] | ensemble3 | 0.832 | 0.611 | 0.678 | 0.776 [0.748, 0.802] | 0.941 [0.929, 0.952] | 0.789 | 0.798 |
| 8 | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v3fundus] | ensemble2 | 0.809 | 0.662 | 0.703 | 0.775 [0.745, 0.803] | 0.940 [0.928, 0.951] | 0.786 | 0.807 |
| 9 | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v3fundus + vit_dinov2_cluster_c_v3fundus] | ensemble3 | 0.822 | 0.627 | 0.690 | 0.774 [0.747, 0.801] | 0.941 [0.929, 0.952] | 0.781 | 0.802 |
| 10 | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v3fundus_518] | ensemble2 | 0.824 | 0.623 | 0.686 | 0.774 [0.747, 0.797] | 0.937 [0.924, 0.949] | 0.780 | 0.792 |
| 11 | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v2 + vit_residual_attn_cluster_c_v3fundus] | ensemble3 | 0.818 | 0.633 | 0.697 | 0.773 [0.744, 0.800] | 0.942 [0.930, 0.953] | 0.790 | 0.800 |
| 12 | vit_residual_attn_cluster_c_v2_518 | single | 0.835 | 0.595 | 0.664 | 0.773 [0.744, 0.799] | 0.925 [0.909, 0.939] | 0.746 | 0.787 |
| 13 | ENS[vit_residual_attn_cluster_c_v2_518 + vit_dinov2_cluster_c_v2] | ensemble2 | 0.833 | 0.599 | 0.656 | 0.773 [0.745, 0.800] | 0.934 [0.920, 0.947] | 0.764 | 0.799 |
| 14 | ENS[vit_residual_attn_cluster_c_v3fundus + vit_residual_attn_cluster_c_v3fundus_518 + vit_dinov2_cluster_c_v2] | ensemble3 | 0.812 | 0.647 | 0.704 | 0.772 [0.741, 0.801] | 0.943 [0.931, 0.953] | 0.787 | 0.792 |
| 15 | ENS[vit_residual_attn_cluster_c_v3fundus_518 + vit_dinov2_cluster_c_v2 + vit_dinov2_cluster_c_v3fundus] | ensemble3 | 0.821 | 0.624 | 0.687 | 0.772 [0.742, 0.800] | 0.941 [0.931, 0.951] | 0.776 | 0.793 |

## cluster_c — recall80

| # | Modelo | Tipo | Recall | Prec. | F1 | F2 [IC95] | AUROC [IC95] | mAP | F2 VAL |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v3fundus + vit_residual_attn_cluster_c_v3fundus_518] | ensemble3 | 0.793 | 0.697 | 0.709 | 0.772 [0.741, 0.800] | 0.941 [0.929, 0.953] | 0.788 | 0.774 |
| 2 | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v3fundus_518] | ensemble2 | 0.797 | 0.684 | 0.697 | 0.772 [0.743, 0.798] | 0.937 [0.924, 0.949] | 0.780 | 0.765 |
| 3 | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v3fundus] | ensemble2 | 0.792 | 0.686 | 0.709 | 0.768 [0.738, 0.798] | 0.940 [0.928, 0.951] | 0.786 | 0.771 |
| 4 | ENS[vit_residual_attn_cluster_c_v3fundus_518 + vit_dinov2_cluster_c_v2 + vit_dinov2_cluster_c_v3fundus] | ensemble3 | 0.791 | 0.687 | 0.698 | 0.768 [0.737, 0.796] | 0.941 [0.931, 0.951] | 0.776 | 0.774 |
| 5 | ENS[vit_residual_attn_cluster_c_v2 + vit_residual_attn_cluster_c_v3fundus + vit_residual_attn_cluster_c_v3fundus_518] | ensemble3 | 0.793 | 0.675 | 0.699 | 0.766 [0.735, 0.795] | 0.941 [0.929, 0.952] | 0.789 | 0.768 |
| 6 | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v3fundus_518 + vit_dinov2_cluster_c_v3fundus] | ensemble3 | 0.782 | 0.700 | 0.700 | 0.764 [0.735, 0.792] | 0.941 [0.929, 0.952] | 0.781 | 0.774 |
| 7 | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v3fundus + vit_dinov2_cluster_c_v3fundus] | ensemble3 | 0.784 | 0.694 | 0.707 | 0.764 [0.733, 0.794] | 0.941 [0.929, 0.952] | 0.781 | 0.776 |
| 8 | ENS[vit_residual_attn_cluster_c_v2_518 + vit_dinov2_cluster_c_v3fundus] | ensemble2 | 0.782 | 0.698 | 0.704 | 0.764 [0.734, 0.793] | 0.940 [0.929, 0.951] | 0.772 | 0.778 |
| 9 | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v3fundus_518 + vit_dinov2_cluster_c_v2] | ensemble3 | 0.788 | 0.676 | 0.698 | 0.763 [0.732, 0.791] | 0.941 [0.929, 0.952] | 0.788 | 0.772 |
| 10 | ENS[vit_residual_attn_cluster_c_v2 + vit_residual_attn_cluster_c_v3fundus_518 + vit_dinov2_cluster_c_v3fundus] | ensemble3 | 0.781 | 0.685 | 0.693 | 0.760 [0.730, 0.789] | 0.939 [0.928, 0.950] | 0.779 | 0.771 |
| 11 | ENS[vit_residual_attn_cluster_c_v3fundus + vit_residual_attn_cluster_c_v3fundus_518 + vit_dinov2_cluster_c_v2] | ensemble3 | 0.783 | 0.679 | 0.693 | 0.760 [0.729, 0.789] | 0.943 [0.931, 0.953] | 0.787 | 0.770 |
| 12 | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v2 + vit_residual_attn_cluster_c_v3fundus_518] | ensemble3 | 0.788 | 0.660 | 0.683 | 0.759 [0.729, 0.786] | 0.940 [0.927, 0.952] | 0.790 | 0.767 |
| 13 | ENS[vit_dinov2_cluster_c_v2 + vit_dinov2_cluster_c_v3fundus] | ensemble2 | 0.785 | 0.669 | 0.684 | 0.759 [0.727, 0.788] | 0.941 [0.930, 0.950] | 0.769 | 0.775 |
| 14 | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v2 + vit_residual_attn_cluster_c_v3fundus] | ensemble3 | 0.782 | 0.672 | 0.693 | 0.757 [0.727, 0.787] | 0.942 [0.930, 0.953] | 0.790 | 0.771 |
| 15 | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v3fundus + vit_dinov2_cluster_c_v2] | ensemble3 | 0.781 | 0.666 | 0.690 | 0.755 [0.723, 0.786] | 0.944 [0.932, 0.955] | 0.792 | 0.775 |

## sup_k12 — fbeta2

| # | Modelo | Tipo | Recall | Prec. | F1 | F2 [IC95] | AUROC [IC95] | mAP | F2 VAL |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ENS[inception_sup_k12_sano_hires448_asl_seed1_tta + inception_sup_k12_sano_hires448_asl_tta] | ensemble2 | 0.771 | 0.507 | 0.593 | 0.698 [0.651, 0.735] | 0.928 [0.903, 0.945] | 0.670 | 0.733 |
| 2 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_tta + inception_sup_k12_sano_hires448_asl_tta] | ensemble2 | 0.749 | 0.544 | 0.604 | 0.697 [0.641, 0.746] | 0.937 [0.911, 0.952] | 0.674 | 0.763 |
| 3 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_tta + inception_sup_k12_sano_hires448_asl_notta] | ensemble2 | 0.739 | 0.549 | 0.601 | 0.691 [0.636, 0.741] | 0.934 [0.903, 0.951] | 0.665 | 0.760 |
| 4 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_notta + inception_sup_k12_sano_hires448_asl_tta] | ensemble2 | 0.738 | 0.545 | 0.601 | 0.689 [0.634, 0.740] | 0.936 [0.910, 0.951] | 0.669 | 0.758 |
| 5 | ENS[inception_sup_k12_sano_hires448_asl_seed1_tta + vit_dinov2_sup_k12_sano_hires518_focal_seed2_notta + inception_sup_k12_sano_hires448_asl_tta] | ensemble3 | 0.743 | 0.533 | 0.582 | 0.689 [0.633, 0.740] | 0.940 [0.916, 0.954] | 0.665 | 0.763 |
| 6 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_notta + inception_sup_k12_sano_hires448_asl_seed1_notta + inception_sup_k12_sano_hires448_asl_notta] | ensemble3 | 0.750 | 0.514 | 0.562 | 0.687 [0.636, 0.732] | 0.937 [0.912, 0.952] | 0.662 | 0.774 |
| 7 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_tta + inception_sup_k12_sano_hires448_asl_seed1_tta + inception_sup_k12_sano_hires448_asl_notta] | ensemble3 | 0.740 | 0.523 | 0.571 | 0.684 [0.631, 0.733] | 0.938 [0.914, 0.953] | 0.668 | 0.763 |
| 8 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_notta + inception_sup_k12_sano_hires448_asl_notta] | ensemble2 | 0.738 | 0.522 | 0.577 | 0.681 [0.627, 0.732] | 0.933 [0.903, 0.950] | 0.660 | 0.758 |
| 9 | inception_sup_k12_sano_hires448_asl_tta | single | 0.740 | 0.514 | 0.590 | 0.681 [0.621, 0.727] | 0.913 [0.881, 0.933] | 0.641 | 0.710 |
| 10 | ENS[inception_sup_k12_sano_hires448_asl_seed1_notta + inception_sup_k12_sano_hires448_asl_tta] | ensemble2 | 0.753 | 0.491 | 0.573 | 0.680 [0.635, 0.717] | 0.926 [0.900, 0.943] | 0.666 | 0.730 |
| 11 | ENS[inception_sup_k12_sano_hires448_asl_seed1_tta + vit_dinov2_sup_k12_sano_hires518_focal_seed2_notta + inception_sup_k12_sano_hires448_asl_notta] | ensemble3 | 0.738 | 0.504 | 0.559 | 0.675 [0.622, 0.724] | 0.939 [0.915, 0.954] | 0.665 | 0.764 |
| 12 | inception_sup_k12_sano_hires448_asl_notta | single | 0.738 | 0.490 | 0.565 | 0.670 [0.622, 0.710] | 0.906 [0.871, 0.927] | 0.630 | 0.707 |
| 13 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_tta + inception_sup_k12_sano_hires448_asl_seed1_tta] | ensemble2 | 0.723 | 0.516 | 0.571 | 0.669 [0.620, 0.723] | 0.930 [0.899, 0.948] | 0.642 | 0.757 |
| 14 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_tta + inception_sup_k12_sano_hires448_asl_seed1_notta + inception_sup_k12_sano_hires448_asl_notta] | ensemble3 | 0.704 | 0.556 | 0.588 | 0.668 [0.610, 0.721] | 0.937 [0.912, 0.952] | 0.664 | 0.749 |
| 15 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_tta + inception_sup_k12_sano_hires448_asl_seed1_notta + inception_sup_k12_sano_hires448_asl_tta] | ensemble3 | 0.699 | 0.559 | 0.593 | 0.666 [0.609, 0.718] | 0.939 [0.915, 0.953] | 0.666 | 0.748 |

## sup_k12 — recall80

| # | Modelo | Tipo | Recall | Prec. | F1 | F2 [IC95] | AUROC [IC95] | mAP | F2 VAL |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ENS[inception_sup_k12_sano_hires448_asl_seed1_notta + inception_sup_k12_sano_hires448_asl_tta] | ensemble2 | 0.769 | 0.567 | 0.593 | 0.718 [0.671, 0.750] | 0.926 [0.900, 0.943] | 0.666 | 0.748 |
| 2 | inception_sup_k12_sano_hires448_asl_tta | single | 0.782 | 0.502 | 0.539 | 0.703 [0.661, 0.735] | 0.913 [0.881, 0.933] | 0.641 | 0.726 |
| 3 | ENS[inception_sup_k12_sano_hires448_asl_seed1_tta + inception_sup_k12_sano_hires448_asl_tta] | ensemble2 | 0.750 | 0.563 | 0.584 | 0.703 [0.648, 0.746] | 0.928 [0.903, 0.945] | 0.670 | 0.747 |
| 4 | inception_sup_k12_sano_hires448_asl_notta | single | 0.779 | 0.502 | 0.539 | 0.701 [0.658, 0.733] | 0.906 [0.871, 0.927] | 0.630 | 0.724 |
| 5 | inception_sup_k12_sano_hires448_asl_seed2_notta | single | 0.778 | 0.477 | 0.532 | 0.691 [0.647, 0.726] | 0.901 [0.870, 0.921] | 0.629 | 0.701 |
| 6 | ENS[inception_sup_k12_sano_hires448_asl_seed1_notta + inception_sup_k12_sano_hires448_asl_notta] | ensemble2 | 0.731 | 0.550 | 0.568 | 0.686 [0.632, 0.733] | 0.923 [0.896, 0.941] | 0.660 | 0.760 |
| 7 | ENS[inception_sup_k12_sano_hires448_asl_seed1_tta + inception_sup_k12_sano_hires448_asl_notta] | ensemble2 | 0.731 | 0.544 | 0.564 | 0.684 [0.630, 0.731] | 0.925 [0.899, 0.943] | 0.660 | 0.760 |
| 8 | inception_sup_k12_sano_hires448_asl_seed2_tta | single | 0.774 | 0.466 | 0.530 | 0.684 [0.638, 0.718] | 0.907 [0.876, 0.926] | 0.640 | 0.692 |
| 9 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_tta + inception_sup_k12_sano_hires448_asl_tta] | ensemble2 | 0.717 | 0.526 | 0.541 | 0.669 [0.633, 0.701] | 0.937 [0.911, 0.952] | 0.674 | 0.762 |
| 10 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_tta + inception_sup_k12_sano_hires448_asl_notta] | ensemble2 | 0.713 | 0.530 | 0.540 | 0.667 [0.631, 0.700] | 0.934 [0.903, 0.951] | 0.665 | 0.763 |
| 11 | ENS[inception_sup_k12_sano_hires448_asl_seed1_tta + vit_dinov2_sup_k12_sano_hires518_focal_seed2_notta + inception_sup_k12_sano_hires448_asl_notta] | ensemble3 | 0.701 | 0.557 | 0.559 | 0.666 [0.628, 0.700] | 0.939 [0.915, 0.954] | 0.665 | 0.768 |
| 12 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_tta + inception_sup_k12_sano_hires448_asl_seed1_tta + inception_sup_k12_sano_hires448_asl_tta] | ensemble3 | 0.698 | 0.560 | 0.565 | 0.666 [0.628, 0.700] | 0.940 [0.917, 0.954] | 0.666 | 0.769 |
| 13 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_tta + inception_sup_k12_sano_hires448_asl_seed1_tta + inception_sup_k12_sano_hires448_asl_notta] | ensemble3 | 0.699 | 0.558 | 0.561 | 0.665 [0.628, 0.700] | 0.938 [0.914, 0.953] | 0.668 | 0.769 |
| 14 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_notta + inception_sup_k12_sano_hires448_asl_notta] | ensemble2 | 0.711 | 0.520 | 0.532 | 0.662 [0.627, 0.696] | 0.933 [0.903, 0.950] | 0.660 | 0.758 |
| 15 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_notta + inception_sup_k12_sano_hires448_asl_tta] | ensemble2 | 0.710 | 0.521 | 0.534 | 0.662 [0.627, 0.695] | 0.936 [0.910, 0.951] | 0.669 | 0.759 |

## Diferencias emparejadas (ensamble elegido en VAL − individual), op fbeta2

| Taxonomía | Ensamble | vs | Métrica | Δ medio | IC95 | P(Δ≤0) |
|---|---|---|---|---|---|---|
| odir7 | ENS[vit_residual_attn_odir7_v2_518 + vit_residual_attn_odir7_v2 + vit_residual_attn_odir7_v3fundus] | vit_residual_attn_odir7_v2_518 | f2 | +0.023 | [+0.008, +0.039] | 0.002 |
| odir7 | ENS[vit_residual_attn_odir7_v2_518 + vit_residual_attn_odir7_v2 + vit_residual_attn_odir7_v3fundus] | vit_residual_attn_odir7_v2_518 | f1 | +0.023 | [+0.009, +0.038] | 0.001 |
| odir7 | ENS[vit_residual_attn_odir7_v2_518 + vit_residual_attn_odir7_v2 + vit_residual_attn_odir7_v3fundus] | vit_residual_attn_odir7_v2_518 | auroc | +0.014 | [+0.008, +0.020] | 0.000 |
| odir7 | ENS[vit_residual_attn_odir7_v2_518 + vit_residual_attn_odir7_v2 + vit_residual_attn_odir7_v3fundus] | vit_residual_attn_odir7_v2_518 | map | +0.033 | [+0.020, +0.046] | 0.000 |
| odir7 | ENS[vit_residual_attn_odir7_v2_518 + vit_residual_attn_odir7_v2 + vit_residual_attn_odir7_v3fundus] | vit_residual_attn_odir7_v2 | f2 | +0.021 | [+0.007, +0.035] | 0.000 |
| odir7 | ENS[vit_residual_attn_odir7_v2_518 + vit_residual_attn_odir7_v2 + vit_residual_attn_odir7_v3fundus] | vit_residual_attn_odir7_v2 | f1 | +0.039 | [+0.025, +0.052] | 0.000 |
| odir7 | ENS[vit_residual_attn_odir7_v2_518 + vit_residual_attn_odir7_v2 + vit_residual_attn_odir7_v3fundus] | vit_residual_attn_odir7_v2 | auroc | +0.014 | [+0.009, +0.019] | 0.000 |
| odir7 | ENS[vit_residual_attn_odir7_v2_518 + vit_residual_attn_odir7_v2 + vit_residual_attn_odir7_v3fundus] | vit_residual_attn_odir7_v2 | map | +0.034 | [+0.019, +0.051] | 0.000 |
| cluster_c | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v3fundus] | vit_residual_attn_cluster_c_v2_518 | f2 | +0.002 | [-0.016, +0.021] | 0.432 |
| cluster_c | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v3fundus] | vit_residual_attn_cluster_c_v2_518 | f1 | +0.039 | [+0.021, +0.058] | 0.000 |
| cluster_c | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v3fundus] | vit_residual_attn_cluster_c_v2_518 | auroc | +0.015 | [+0.006, +0.025] | 0.000 |
| cluster_c | ENS[vit_residual_attn_cluster_c_v2_518 + vit_residual_attn_cluster_c_v3fundus] | vit_residual_attn_cluster_c_v2_518 | map | +0.040 | [+0.025, +0.057] | 0.000 |
| sup_k12 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_notta + inception_sup_k12_sano_hires448_asl_seed1_notta + inception_sup_k12_sano_hires448_asl_notta] | vit_dinov2_sup_k12_sano_hires518_focal_seed2_tta | f2 | +0.061 | [+0.012, +0.111] | 0.004 |
| sup_k12 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_notta + inception_sup_k12_sano_hires448_asl_seed1_notta + inception_sup_k12_sano_hires448_asl_notta] | vit_dinov2_sup_k12_sano_hires518_focal_seed2_tta | f1 | +0.057 | [+0.022, +0.093] | 0.000 |
| sup_k12 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_notta + inception_sup_k12_sano_hires448_asl_seed1_notta + inception_sup_k12_sano_hires448_asl_notta] | vit_dinov2_sup_k12_sano_hires518_focal_seed2_tta | auroc | +0.030 | [+0.010, +0.056] | 0.001 |
| sup_k12 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_notta + inception_sup_k12_sano_hires448_asl_seed1_notta + inception_sup_k12_sano_hires448_asl_notta] | vit_dinov2_sup_k12_sano_hires518_focal_seed2_tta | map | +0.067 | [+0.026, +0.110] | 0.000 |
| sup_k12 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_notta + inception_sup_k12_sano_hires448_asl_seed1_notta + inception_sup_k12_sano_hires448_asl_notta] | inception_sup_k12_sano_hires448_asl_tta | f2 | +0.007 | [-0.044, +0.051] | 0.370 |
| sup_k12 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_notta + inception_sup_k12_sano_hires448_asl_seed1_notta + inception_sup_k12_sano_hires448_asl_notta] | inception_sup_k12_sano_hires448_asl_tta | f1 | -0.026 | [-0.069, +0.012] | 0.905 |
| sup_k12 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_notta + inception_sup_k12_sano_hires448_asl_seed1_notta + inception_sup_k12_sano_hires448_asl_notta] | inception_sup_k12_sano_hires448_asl_tta | auroc | +0.024 | [+0.010, +0.040] | 0.001 |
| sup_k12 | ENS[vit_dinov2_sup_k12_sano_hires518_focal_seed2_notta + inception_sup_k12_sano_hires448_asl_seed1_notta + inception_sup_k12_sano_hires448_asl_notta] | inception_sup_k12_sano_hires448_asl_tta | map | +0.020 | [-0.008, +0.045] | 0.058 |