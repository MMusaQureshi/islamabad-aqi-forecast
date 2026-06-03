\# Direct Multi-Horizon AQI Forecasting - Random Forest V2



\## Date

2026-06-03



\## Forecasting Method

Direct multi-horizon forecasting



The model directly predicts:

\- Day 1 AQI

\- Day 2 AQI

\- Day 3 AQI



This avoids recursive error accumulation.



\## Best Model

Random Forest



\## Training Rows

163



\## Overall Test Metrics

| Metric | Value |

|---|---:|

| Avg Test RMSE | 24.69 |

| Avg Test MAE | 14.91 |

| Avg Test R² | -0.20 |



\## Day-wise Test Metrics

| Horizon | Test RMSE | Test MAE | Test R² |

|---|---:|---:|---:|

| Day 1 | 17.924 | 10.306 | 0.114 |

| Day 2 | 25.428 | 15.406 | -0.285 |

| Day 3 | 30.720 | 19.015 | -0.423 |



\## Validation Comparison

| Model | Avg Validation RMSE | Avg Validation MAE | Avg Validation R² |

|---|---:|---:|---:|

| Random Forest | 33.694 | 25.726 | -0.625 |

| Ridge Regression | 37.451 | 30.808 | -1.170 |

| TensorFlow MLP | 47.685 | 39.761 | -2.126 |



\## Notes

This version improved test MAE compared to the previous direct model.



Previous direct model:

\- Avg Test RMSE: 26.34

\- Avg Test MAE: 17.98

\- Avg Test R²: -0.25



Current model:

\- Avg Test RMSE: 24.69

\- Avg Test MAE: 14.91

\- Avg Test R²: -0.20



The model is better on RMSE, MAE, and R², but R² is still slightly negative overall. Day 1 forecasting is the strongest, while Day 2 and Day 3 still need improvement.

