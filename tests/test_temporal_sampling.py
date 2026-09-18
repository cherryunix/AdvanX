#!/usr/bin/env python3

import unittest

import numpy as np

from advanx.temporal import periodic_select_pattern, target_source_indices


class TargetSourceIndicesTest(unittest.TestCase):
    def test_100_to_24_is_uniform_and_unique(self) -> None:
        indices = target_source_indices(1001, 100.0, 24.0)

        self.assertEqual(len(indices), 241)
        np.testing.assert_array_equal(indices[:7], [0, 4, 8, 13, 17, 21, 25])
        self.assertEqual(indices[-1], 1000)
        self.assertEqual(set(np.diff(indices).tolist()), {4, 5})

    def test_2997_to_24_only_drops_frames(self) -> None:
        indices = target_source_indices(301, 30000 / 1001, 24.0)

        self.assertEqual(len(indices), 241)
        self.assertTrue(np.all(np.diff(indices) > 0))
        self.assertEqual(set(np.diff(indices).tolist()), {1, 2})

    def test_same_rate_keeps_every_frame(self) -> None:
        np.testing.assert_array_equal(
            target_source_indices(241, 24.0, 24.0), np.arange(241)
        )

    def test_upsampling_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            target_source_indices(100, 23.976, 24.0)

    def test_software_decode_pattern_matches_exact_100_to_24_indices(self) -> None:
        period, offsets = periodic_select_pattern(100.0, 24.0)
        actual = np.array(
            [cycle * period + offset for cycle in range(10) for offset in offsets]
        )
        expected = target_source_indices(10 * period, 100.0, 24.0)

        self.assertEqual(period, 25)
        self.assertEqual(offsets, (0, 4, 8, 13, 17, 21))
        np.testing.assert_array_equal(actual, expected)

    def test_software_decode_pattern_handles_30_to_24(self) -> None:
        self.assertEqual(periodic_select_pattern(30.0, 24.0), (5, (0, 1, 3, 4)))


if __name__ == "__main__":
    unittest.main()
