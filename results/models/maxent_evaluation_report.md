# MaxEntScan Upgrade Evaluation Report

## Performance Summary

| Evaluation Set | Model | Features | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Internal (KD1) | RF V2 (Additive PWM) | 23 | 74.81% | 0.6611 | 0.7423 | 0.9080 |
| Internal (KD1) | RF V3 (MaxEntScan) | 23 | 75.79% | 0.6203 | 0.7419 | 0.9055 |
| External (KD2) | RF V2 (Additive PWM) | 23 | 76.82% | 0.7003 | 0.7672 | 0.9168 |
| External (KD2) | RF V3 (MaxEntScan) | 23 | 79.41% | 0.6897 | 0.7846 | 0.9187 |

## Best Hyperparameters

```python
{'max_depth': 20, 'n_estimators': 100}
```
