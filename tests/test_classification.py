import unittest

import numpy as np

from advanx.classification import best_upper_threshold


class ClassificationTest(unittest.TestCase):
    def test_best_upper_threshold_separates_low_positive_values(self) -> None:
        threshold, metrics = best_upper_threshold(
            np.array([1.0, 2.0, 8.0, 9.0]),
            np.array([True, True, False, False]),
        )
        self.assertEqual(threshold, 5.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["false_positive"], 0)
        self.assertEqual(metrics["false_negative"], 0)


if __name__ == "__main__":
    unittest.main()
