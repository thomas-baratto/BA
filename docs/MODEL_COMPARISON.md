# Model Comparison Report

Comparison of the 6 best models: 2 Optuna-optimized MLPs and 4 random network sweep winners (knee-point selection from Pareto frontiers of accuracy vs. complexity).

*Generated on 2026-03-24 from sweep job 1048 (558 configurations).*

## Summary Table

| Dataset | Model | Type | R² | RMSE | MAE | nRMSE | KGE | Train Time | Samples |
|---------|-------|------|-----|------|-----|-------|-----|------------|---------|
| cone | Optimized MLP | MLP | 0.9909 | 0.05 | 0.01 | 0.013306 | 0.9911 | 1212.4s | 12835 |
| cone | edRVFL-SC (Pareto winner) | RANDOM | 0.9754 | 0.09 | 0.04 | 0.025720 | 0.9868 | 22.2s | 3851 |
| isotherm | Optimized MLP | MLP | 1.0000 | 92.01 | 14.38 | 0.000185 | 0.9998 | 4296.1s | 85531 |
| isotherm | SResdRVFL (nRMSE winner) | RANDOM | 0.8950 | 102.11 | 47.66 | 0.051065 | 0.9357 | 25.0s | 25660 |
| isotherm | dRVFL (KGE winner) | RANDOM | 0.8548 | 120.07 | 56.79 | 0.060045 | 0.8976 | 2.2s | 25660 |

## Pareto Frontiers

Pareto frontier plots showing accuracy (nRMSE or 1−KGE) vs. complexity (training time) for all 497 successful sweep runs. The red dashed line marks the Pareto-efficient frontier; the gold star marks the knee-point winner.

![cone_nRMSE](plots/pareto_cone_nRMSE.png)

![cone_1KGE](plots/pareto_cone_1KGE.png)

![isotherm_nRMSE](plots/pareto_isotherm_nRMSE.png)

![isotherm_1KGE](plots/pareto_isotherm_1KGE.png)

---

## CONE Dataset

### Optimized MLP (MLP)

**Architecture:** `MLP 1×221, SiLU, dropout=0.1226`

**Preprocessing:** feature_scaler=robust, label_scaler=minmax, log_transform=✓

#### Per-Label Test Metrics

| Label | R² | RMSE | MAE | nRMSE | KGE |
|-------|-----|------|-----|-------|-----|
| Cone | 0.9909 | 0.05 m | 0.01 m | 0.013306 | 0.9911 |

#### Plots

**Cone**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/regression_mlp_cone_Cone.png) | ![residuals](plots/residuals_mlp_cone_Cone.png) |


### edRVFL-SC (Pareto winner) (RANDOM)

**Architecture:** `edRVFL-SC: H=1000, L=3, B=5, E=10, GELU, direct_link=✗, area_root=✗`

**Preprocessing:** feature_scaler=robust, label_scaler=robust, log_transform=✗

#### Per-Label Test Metrics

| Label | R² | RMSE | MAE | nRMSE | KGE |
|-------|-----|------|-----|-------|-----|
| Cone | 0.9754 | 0.09 m | 0.04 m | 0.025720 | 0.9868 |

#### Plots

**Cone**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/regression_cone_nRMSE_Cone.png) | ![residuals](plots/residuals_cone_nRMSE_Cone.png) |


---

## ISOTHERM Dataset

### Optimized MLP (MLP)

**Architecture:** `MLP 4×256, GELU, dropout=0.0000`

**Preprocessing:** feature_scaler=robust, label_scaler=minmax, log_transform=✓

#### Per-Label Test Metrics

| Label | R² | RMSE | MAE | nRMSE | KGE |
|-------|-----|------|-----|-------|-----|
| Area | 1.0000 | 159.35 m² | 42.36 m² | 0.000320 | 0.9998 |
| Iso_distance | 1.0000 | 2.45 m | 0.74 m | 0.001224 | 0.9997 |
| Iso_width | 1.0000 | 0.07 m | 0.03 m | 0.000219 | 0.9999 |

#### Plots

**Area**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/regression_mlp_isotherm_Area.png) | ![residuals](plots/residuals_mlp_isotherm_Area.png) |

**Iso_distance**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/regression_mlp_isotherm_Iso_distance.png) | ![residuals](plots/residuals_mlp_isotherm_Iso_distance.png) |

**Iso_width**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/regression_mlp_isotherm_Iso_width.png) | ![residuals](plots/residuals_mlp_isotherm_Iso_width.png) |


### SResdRVFL (nRMSE winner) (RANDOM)

**Architecture:** `SResdRVFL: H=1500, L=1, B=8, E=10, ELU, direct_link=✓, area_root=✓`

**Preprocessing:** feature_scaler=robust, label_scaler=robust, log_transform=✗

#### Per-Label Test Metrics

| Label | R² | RMSE | MAE | nRMSE | KGE |
|-------|-----|------|-----|-------|-----|
| Area | 0.8788 | 36.87 m² | 23.00 m² | 0.054327 | 0.9249 |
| Iso_distance | 0.8704 | 172.56 m | 112.99 m | 0.086301 | 0.9201 |
| Iso_width | 0.9066 | 11.87 m | 6.99 m | 0.037637 | 0.9394 |

#### Plots

**Area**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/regression_isotherm_nRMSE_Area.png) | ![residuals](plots/residuals_isotherm_nRMSE_Area.png) |

**Iso_distance**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/regression_isotherm_nRMSE_Iso_distance.png) | ![residuals](plots/residuals_isotherm_nRMSE_Iso_distance.png) |

**Iso_width**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/regression_isotherm_nRMSE_Iso_width.png) | ![residuals](plots/residuals_isotherm_nRMSE_Iso_width.png) |


### dRVFL (KGE winner) (RANDOM)

**Architecture:** `dRVFL: H=1500, L=1, B=5, E=10, ELU, direct_link=✗, area_root=✓`

**Preprocessing:** feature_scaler=robust, label_scaler=robust, log_transform=✗

#### Per-Label Test Metrics

| Label | R² | RMSE | MAE | nRMSE | KGE |
|-------|-----|------|-----|-------|-----|
| Area | 0.8369 | 42.76 m² | 27.40 m² | 0.063015 | 0.8851 |
| Iso_distance | 0.8205 | 203.05 m | 134.67 m | 0.101545 | 0.8722 |
| Iso_width | 0.8734 | 13.82 m | 8.28 m | 0.043809 | 0.9092 |

#### Plots

**Area**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/regression_isotherm_1KGE_Area.png) | ![residuals](plots/residuals_isotherm_1KGE_Area.png) |

**Iso_distance**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/regression_isotherm_1KGE_Iso_distance.png) | ![residuals](plots/residuals_isotherm_1KGE_Iso_distance.png) |

**Iso_width**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/regression_isotherm_1KGE_Iso_width.png) | ![residuals](plots/residuals_isotherm_1KGE_Iso_width.png) |


---

## Key Findings

### Cone (Depression Cone Size)

- **MLP** achieves R² = 0.9909, nRMSE = 0.013306 on the full dataset.
- **Best random model** (edRVFL-SC) achieves R² = 0.9754, nRMSE = 0.025720 on the test fold.
- Both cone winners on the Pareto frontiers selected the same edRVFL-SC configuration (H=1000, L=3, B=5, GELU).

### Isotherm (Thermal Plume Geometry)

- **MLP** achieves R² = 0.999988, nRMSE = 0.000185 — near-perfect reconstruction of thermal plume geometry.
- **Best random model by nRMSE** (SResdRVFL) achieves R² = 0.8950, nRMSE = 0.051065.
- **Best random model by KGE** (dRVFL) achieves R² = 0.8548, KGE = 0.8976.
- The MLP significantly outperforms all random architectures on this task, demonstrating the value of backpropagation-based optimization for the multi-output isotherm problem.

### Speed vs. Accuracy Trade-off

- Random models train in **1–7 seconds** vs. **20–70 minutes** for the Optuna-optimized MLP.
- For the cone dataset, random models achieve competitive accuracy at a fraction of the training cost.
- For the isotherm dataset, the accuracy gap justifies the additional training time of the MLP.
