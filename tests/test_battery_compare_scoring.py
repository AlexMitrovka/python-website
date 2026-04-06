import math
import unittest

from battery_compare_scoring import BASE_WEIGHTS, compute_compare_scores, normalize_higher_better, normalize_lower_better


class CompareScoringTests(unittest.TestCase):
    def test_normalize_higher_better(self) -> None:
        self.assertAlmostEqual(normalize_higher_better(100.0, 80.0, 120.0), 55.0, places=6)
        self.assertAlmostEqual(normalize_higher_better(120.0, 80.0, 120.0), 100.0, places=6)
        self.assertAlmostEqual(normalize_higher_better(80.0, 80.0, 120.0), 10.0, places=6)

    def test_normalize_lower_better(self) -> None:
        self.assertAlmostEqual(normalize_lower_better(100.0, 80.0, 120.0), 55.0, places=6)
        self.assertAlmostEqual(normalize_lower_better(80.0, 80.0, 120.0), 100.0, places=6)
        self.assertAlmostEqual(normalize_lower_better(120.0, 80.0, 120.0), 10.0, places=6)

    def test_equal_values_become_100(self) -> None:
        pack = compute_compare_scores(
            [100.0, 100.0],
            [8.0, 8.0],
            [50.0, 50.0],
            [0.1, 0.1],
            weights=BASE_WEIGHTS,
        )
        self.assertEqual(pack["Score mAh"], [100.0, 100.0])
        self.assertEqual(pack["Score Wh"], [100.0, 100.0])
        self.assertEqual(pack["Score IR DC"], [100.0, 100.0])
        self.assertEqual(pack["Score ΔV"], [100.0, 100.0])
        self.assertEqual(pack["Final Score"], [100.0, 100.0])

    def test_nan_values_are_neutral(self) -> None:
        pack = compute_compare_scores(
            [100.0, math.nan],
            [8.0, 7.0],
            [50.0, math.nan],
            [0.1, math.nan],
            weights=BASE_WEIGHTS,
        )
        self.assertAlmostEqual(pack["Score mAh"][1], 50.0, places=6)
        self.assertAlmostEqual(pack["Score IR DC"][1], 50.0, places=6)
        self.assertAlmostEqual(pack["Score ΔV"][1], 50.0, places=6)

    def test_final_score_formula(self) -> None:
        pack = compute_compare_scores(
            [2000.0, 2500.0],
            [7.0, 9.0],
            [90.0, 60.0],
            [0.25, 0.12],
            weights=BASE_WEIGHTS,
        )
        # Другий елемент найкращий за всіма осями => 100
        self.assertAlmostEqual(pack["Final Score"][1], 100.0, places=6)
        # Перший найгірший за всіма осями => 10 (через шкалу 10..100)
        self.assertAlmostEqual(pack["Final Score"][0], 10.0, places=6)

    def test_best_battery_near_100(self) -> None:
        pack = compute_compare_scores(
            [2400.0, 2450.0, 2500.0],
            [8.5, 8.7, 9.0],
            [85.0, 70.0, 55.0],
            [0.24, 0.18, 0.12],
            weights=BASE_WEIGHTS,
        )
        self.assertGreaterEqual(pack["Final Score"][2], 99.0)

    def test_worst_battery_near_10(self) -> None:
        pack = compute_compare_scores(
            [2500.0, 2450.0, 2400.0],
            [9.0, 8.7, 8.5],
            [55.0, 70.0, 85.0],
            [0.12, 0.18, 0.24],
            weights=BASE_WEIGHTS,
        )
        self.assertLessEqual(pack["Final Score"][2], 11.0)
        self.assertGreaterEqual(pack["Final Score"][2], 9.0)


if __name__ == "__main__":
    unittest.main()
