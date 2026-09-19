import unittest

import numpy as np

from advanx.segments import fill_short_gaps, split_run, true_runs


class SegmentUtilitiesTest(unittest.TestCase):
    def test_true_runs_are_half_open(self) -> None:
        self.assertEqual(
            true_runs(np.array([False, True, True, False, True])),
            [(1, 3), (4, 5)],
        )

    def test_short_gaps_fill_but_blocked_cuts_do_not(self) -> None:
        mask = np.array([True, True, False, False, True, True])
        np.testing.assert_array_equal(
            fill_short_gaps(mask, 2),
            [True, True, True, True, True, True],
        )
        blockers = np.array([False, False, False, True, False, False])
        np.testing.assert_array_equal(fill_short_gaps(mask, 2, blockers), mask)

    def test_long_run_splits_without_overlap_or_loss(self) -> None:
        intervals = split_run(10, 1010, target_frames=300, max_frames=400, min_frames=200)
        self.assertEqual(intervals[0][0], 10)
        self.assertEqual(intervals[-1][1], 1010)
        self.assertTrue(all(left[1] == right[0] for left, right in zip(intervals, intervals[1:])))
        self.assertTrue(all(200 <= end - start <= 400 for start, end in intervals))


if __name__ == "__main__":
    unittest.main()
