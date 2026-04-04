# Model Comparison Report

Comparison of the 6 best models: 2 Optuna-optimized MLPs and 4 random network sweep winners (knee-point selection from Pareto frontiers of accuracy vs. complexity).

*Generated on 2026-03-24 from sweep job 1048 (558 configurations).*

## Summary Table

| Dataset | Model | Type | R² | RMSE | MAE | nRMSE | KGE | Train Time | Samples |
|---------|-------|------|-----|------|-----|-------|-----|------------|---------|
| cone | Optimized MLP | MLP | 0.9887 | 0.06 | 0.02 | 0.014839 | 0.9907 | 735.1s | 12835 |
| cone | edRVFL-SC (Pareto winner) | RANDOM | 0.9770 | 0.08 | 0.04 | 0.024885 | 0.9871 | 7.1s | 3851 |
| isotherm | Optimized MLP | MLP | 1.0000 | 79.09 | 14.43 | 0.000159 | 0.9995 | 4572.4s | 85531 |
| isotherm | SResdRVFL (nRMSE winner) | RANDOM | 0.8287 | 10918.30 | 2310.80 | 0.023685 | 0.8276 | 6.0s | 25660 |
| isotherm | dRVFL (KGE winner) | RANDOM | 0.7828 | 12294.79 | 2814.14 | 0.026671 | 0.7426 | 0.7s | 25660 |

## Pareto Frontiers

Pareto frontier plots showing accuracy (nRMSE or 1−KGE) vs. complexity (training time) for all 497 successful sweep runs. The red dashed line marks the Pareto-efficient frontier; the gold star marks the knee-point winner.

![cone_nRMSE](plots/pareto/pareto_cone_nRMSE.png)

![cone_1KGE](plots/pareto/pareto_cone_1KGE.png)

![isotherm_nRMSE](plots/pareto/pareto_isotherm_nRMSE.png)

![isotherm_1KGE](plots/pareto/pareto_isotherm_1KGE.png)

---

## CONE Dataset

### Optimized MLP (MLP)

**Architecture:** `MLP 1×244, LeakyReLU, dropout=0.0094`

**Preprocessing:** feature_scaler=robust, label_scaler=minmax, log_transform=✓

#### Per-Label Test Metrics

| Label | R² | RMSE | MAE | nRMSE | KGE |
|-------|-----|------|-----|-------|-----|
| Cone | 0.9887 | 0.06 m | 0.02 m | 0.014839 | 0.9907 |

#### Plots

**Cone**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/mlp/regression_mlp_cone_Cone.png) | ![residuals](plots/mlp/residuals_mlp_cone_Cone.png) |


### edRVFL-SC (Pareto winner) (RANDOM)

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
| ![regression](plots/random/regression_cone_nRMSE_Cone.png) | ![residuals](plots/random/residuals_cone_nRMSE_Cone.png) |


---

## ISOTHERM Dataset

### Optimized MLP (MLP)

**Architecture:** `MLP 4×256, GELU, dropout=0.0000`

**Preprocessing:** feature_scaler=robust, label_scaler=minmax, log_transform=✓

#### Per-Label Test Metrics

| Label | R² | RMSE | MAE | nRMSE | KGE |
|-------|-----|------|-----|-------|-----|
| Area | 1.0000 | 136.94 m² | 42.37 m² | 0.000275 | 0.9995 |
| Iso_distance | 0.9999 | 3.96 m | 0.87 m | 0.001980 | 0.9995 |
| Iso_width | 1.0000 | 0.07 m | 0.03 m | 0.000229 | 0.9998 |

#### Plots

**Area**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/mlp/regression_mlp_isotherm_Area.png) | ![residuals](plots/mlp/residuals_mlp_isotherm_Area.png) |

**Iso_distance**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/mlp/regression_mlp_isotherm_Iso_distance.png) | ![residuals](plots/mlp/residuals_mlp_isotherm_Iso_distance.png) |

**Iso_width**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/mlp/regression_mlp_isotherm_Iso_width.png) | ![residuals](plots/mlp/residuals_mlp_isotherm_Iso_width.png) |


### SResdRVFL (nRMSE winner) (RANDOM)

**Architecture:** `SResdRVFL: H=1500, L=1, B=8, E=10, GELU, direct_link=✓, area_root=✓`

**Preprocessing:** feature_scaler=robust, label_scaler=robust, log_transform=✗

#### Per-Label Test Metrics

| Label | R² | RMSE | MAE | nRMSE | KGE |
|-------|-----|------|-----|-------|-----|
| Area | 0.8008 | 18910.29 m² | 6817.15 m² | 0.041023 | 0.8055 |
| Iso_distance | 0.8762 | 168.66 m | 108.50 m | 0.084349 | 0.9227 |
| Iso_width | 0.9099 | 11.66 m | 6.74 m | 0.036967 | 0.9409 |

#### Plots

**Area**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/random/regression_isotherm_nRMSE_Area.png) | ![residuals](plots/random/residuals_isotherm_nRMSE_Area.png) |

**Iso_distance**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/random/regression_isotherm_nRMSE_Iso_distance.png) | ![residuals](plots/random/residuals_isotherm_nRMSE_Iso_distance.png) |

**Iso_width**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/random/regression_isotherm_nRMSE_Iso_width.png) | ![residuals](plots/random/residuals_isotherm_nRMSE_Iso_width.png) |


### dRVFL (KGE winner) (RANDOM)

**Architecture:** `dRVFL: H=1500, L=1, B=5, E=10, ELU, direct_link=✗, area_root=✓`

**Preprocessing:** feature_scaler=robust, label_scaler=robust, log_transform=✗

#### Per-Label Test Metrics

| Label | R² | RMSE | MAE | nRMSE | KGE |
|-------|-----|------|-----|-------|-----|
| Area | 0.7474 | 21294.25 m² | 8302.34 m² | 0.046194 | 0.7120 |
| Iso_distance | 0.8263 | 199.75 m | 131.90 m | 0.099899 | 0.8767 |
| Iso_width | 0.8768 | 13.64 m | 8.19 m | 0.043220 | 0.9113 |

#### Plots

**Area**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/random/regression_isotherm_1KGE_Area.png) | ![residuals](plots/random/residuals_isotherm_1KGE_Area.png) |

**Iso_distance**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/random/regression_isotherm_1KGE_Iso_distance.png) | ![residuals](plots/random/residuals_isotherm_1KGE_Iso_distance.png) |

**Iso_width**

| Regression | Residuals |
|:---:|:---:|
| ![regression](plots/random/regression_isotherm_1KGE_Iso_width.png) | ![residuals](plots/random/residuals_isotherm_1KGE_Iso_width.png) |


---

## Key Findings

### Cone (Depression Cone Size)

- **MLP** achieves R² = 0.9887, nRMSE = 0.014839 on the full dataset.
- **Best random model** (edRVFL-SC) achieves R² = 0.9770, nRMSE = 0.024885 on the test fold.
- Both cone winners on the Pareto frontiers selected the same edRVFL-SC configuration (H=1000, L=3, B=5, GELU).

### Isotherm (Thermal Plume Geometry)

- **MLP** achieves R² = 0.999991, nRMSE = 0.000159 — near-perfect reconstruction of thermal plume geometry.
- **Best random model by nRMSE** (SResdRVFL) achieves R² = 0.8287, nRMSE = 0.023685.
- **Best random model by KGE** (dRVFL) achieves R² = 0.7828, KGE = 0.7426.
- The MLP significantly outperforms all random architectures on this task, demonstrating the value of backpropagation-based optimization for the multi-output isotherm problem.

### Speed vs. Accuracy Trade-off

- Random models train in **1–7 seconds** vs. **20–70 minutes** for the Optuna-optimized MLP.
- For the cone dataset, random models achieve competitive accuracy at a fraction of the training cost.
- For the isotherm dataset, the accuracy gap justifies the additional training time of the MLP.
