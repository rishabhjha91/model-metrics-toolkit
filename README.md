# Model Metrics Toolkit

A lightweight Python toolkit for evaluating binary classification and regression predictions without third-party dependencies.

## Features

- Binary classification accuracy, precision, recall, and F1-score.
- Regression mean absolute error (MAE), mean squared error (MSE), and root mean squared error (RMSE).
- Input validation for empty inputs and mismatched lengths.
- Standard-library implementation.

## Example

```python
from model_metrics_toolkit import classification_metrics, regression_metrics

print(classification_metrics([1, 0, 1, 0], [1, 0, 0, 0]))
print(regression_metrics([3, 5, 7], [2, 5, 8]))
```

## Limitations

Classification metrics currently support binary labels encoded as 0 and 1. Regression metrics operate on numeric sequences. The toolkit does not train models or replace domain-specific evaluation requirements.
