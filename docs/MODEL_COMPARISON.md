# Model Comparison Report

Comparison of the 6 best models: 2 Optuna-optimized MLPs and 4 random network sweep winners (knee-point selection from Pareto frontiers of accuracy vs. complexity).

*Generated on 2026-03-24 from sweep job 1048 (558 configurations).*

## Summary Table

| Dataset | Model | Type | R² | RMSE | MAE | nRMSE | KGE | Train Time | Samples |
|---------|-------|------|-----|------|-----|-------|-----|------------|---------|
| cone | Optimized MLP | MLP | 0.9909 | 0.05 | 0.01 | 0.013306 | 0.9911 | 1212.4s | 12835 |
| cone | edRVFL-SC (nRMSE winner) | RANDOM | 0.9770 | 0.08 | 0.04 | 0.024885 | 0.9871 | 6.9s | 3851 |
| cone | edRVFL-SC (KGE winner) | RANDOM | 0.9770 | 0.08 | 0.04 | 0.024885 | 0.9871 | 6.9s | 3851 |
| isotherm | Optimized MLP | MLP | 1.0000 | 92.01 | 14.38 | 0.000185 | 0.9998 | 4296.1s | 85531 |
| isotherm | SResdRVFL (nRMSE winner) | RANDOM | 0.8998 | 99.74 | 46.60 | 0.049879 | 0.9400 | 5.9s | 25660 |
| isotherm | dRVFL (KGE winner) | RANDOM | 0.8596 | 118.09 | 55.62 | 0.059058 | 0.9013 | 1.0s | 25660 |

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


### edRVFL-SC (nRMSE winner) (RANDOM)

**Architecture:** `edRVFL-SC: H=1000, L=3, B=5, E=10, GELU, direct_link=✗, area_root=✗`

**Preprocessing:** feature_scaler=robust, label_scaler=robust, log_transform=✗

#### Per-Label Test Metrics

| Label | R² | RMSE | MAE | nRMSE | KGE |
|-------|-----|------|-----|-------|-----|
| Cone | 0.9770 | 0.08 m | 0.04 m | 0.024885 | 0.9871 |

#### Plots

**Cone**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/regression_cone_nRMSE_Cone.png) | ![residuals](plots/residuals_cone_nRMSE_Cone.png) |


### edRVFL-SC (KGE winner) (RANDOM)

**Architecture:** `edRVFL-SC: H=1000, L=3, B=5, E=10, GELU, direct_link=✗, area_root=✗`

**Preprocessing:** feature_scaler=robust, label_scaler=robust, log_transform=✗

#### Per-Label Test Metrics

| Label | R² | RMSE | MAE | nRMSE | KGE |
|-------|-----|------|-----|-------|-----|
| Cone | 0.9770 | 0.08 m | 0.04 m | 0.024885 | 0.9871 |

#### Plots

**Cone**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/regression_cone_1KGE_Cone.png) | ![residuals](plots/residuals_cone_1KGE_Cone.png) |


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
| Area | 0.8839 | 36.09 m² | 22.47 m² | 0.053178 | 0.9292 |
| Iso_distance | 0.8764 | 168.53 m | 110.34 m | 0.084282 | 0.9253 |
| Iso_width | 0.9078 | 11.79 m | 6.97 m | 0.037385 | 0.9413 |

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
| Area | 0.8440 | 41.82 m² | 26.75 m² | 0.061632 | 0.8905 |
| Iso_distance | 0.8263 | 199.75 m | 131.90 m | 0.099899 | 0.8767 |
| Iso_width | 0.8768 | 13.64 m | 8.19 m | 0.043220 | 0.9113 |

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
- **Best random model** (edRVFL-SC) achieves R² = 0.9770, nRMSE = 0.024885 on the test fold.
- Both cone winners on the Pareto frontiers selected the same edRVFL-SC configuration (H=1000, L=3, B=5, GELU).

### Isotherm (Thermal Plume Geometry)

- **MLP** achieves R² = 0.999988, nRMSE = 0.000185 — near-perfect reconstruction of thermal plume geometry.
- **Best random model by nRMSE** (SResdRVFL) achieves R² = 0.8998, nRMSE = 0.049879.
- **Best random model by KGE** (dRVFL) achieves R² = 0.8596, KGE = 0.9013.
- The MLP significantly outperforms all random architectures on this task, demonstrating the value of backpropagation-based optimization for the multi-output isotherm problem.

### Speed vs. Accuracy Trade-off

- Random models train in **1–7 seconds** vs. **20–70 minutes** for the Optuna-optimized MLP.
- For the cone dataset, random models achieve competitive accuracy at a fraction of the training cost.
- For the isotherm dataset, the accuracy gap justifies the additional training time of the MLP.
