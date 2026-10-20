import unittest

from model_metrics_toolkit import classification_metrics, regression_metrics


class MetricsTests(unittest.TestCase):
    def test_classification_metrics(self):
        result = classification_metrics([1, 0, 1, 0], [1, 0, 0, 0])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 2 / 3)

    def test_regression_metrics(self):
        result = regression_metrics([3, -0.5, 2, 7], [2.5, 0, 2, 8])
        self.assertAlmostEqual(result["mae"], 0.5)
        self.assertAlmostEqual(result["mse"], 0.375)
        self.assertAlmostEqual(result["rmse"], 0.375**0.5)

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            regression_metrics([1, 2], [1])

    def test_empty_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            classification_metrics([], [])


if __name__ == "__main__":
    unittest.main()
