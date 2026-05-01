import unittest
import numpy as np
from awg_controller.src.awg_control import RFConverter, AODSettings
from atommovr.utils.Move import Move
from atommovr.utils.core import PhysicalParams


class TestAWGControl(unittest.TestCase):
    def setUp(self):
        self.settings = AODSettings(
            f_min_v=80e6,
            f_max_v=89e6,
            f_min_h=80e6,
            f_max_h=89e6,
            grid_rows=10,
            grid_cols=10
        )
        # AOD_speed 0.1 um/us = 0.1 m/s
        # spacing 5e-6 m = 5 um
        self.params = PhysicalParams(
            AOD_speed=0.1, 
            spacing=5e-6 
        )
        self.converter = RFConverter(self.settings, self.params)

    def test_grid_to_freq(self):
        self.assertEqual(self.converter._row_to_freq(0), 80e6)
        self.assertEqual(self.converter._col_to_freq(0), 80e6)
        self.assertEqual(self.converter._row_to_freq(1), 81e6)
        self.assertEqual(self.converter._col_to_freq(9), 89e6)

    def test_convert_single_move(self):
        move = Move(0, 0, 1, 1)
        batch = self.converter.convert_moves([move])

        expected_duration = 5e-6 / 0.1
        self.assertAlmostEqual(batch.total_duration_s, expected_duration)

        expected_ramps = len(self.converter.core_map[0]) + len(self.converter.core_map[1])
        self.assertEqual(len(batch.ramps), expected_ramps)

        moved_row = [
            r
            for r in batch.ramps
            if r.channel == 0 and r.f_start == self.converter._row_to_freq(0)
        ][0]
        moved_col = [
            r
            for r in batch.ramps
            if r.channel == 1 and r.f_start == self.converter._col_to_freq(0)
        ][0]
        self.assertEqual(moved_row.f_end, self.converter._row_to_freq(1))
        self.assertEqual(moved_col.f_end, self.converter._col_to_freq(1))

    def test_batch_duration(self):
        moves = [Move(0, 0, 0, 1), Move(1, 0, 1, 2)]
        with self.assertRaisesRegex(ValueError, "Conflicting column targets"):
            self.converter.convert_moves(moves)

    def test_empty_batch(self):
        batch = self.converter.convert_moves([])
        holding = self.converter.holding_config()
        self.assertEqual(len(batch.ramps), len(holding.ramps))
        self.assertEqual(batch.total_duration_s, 0.0)


if __name__ == '__main__':
    unittest.main()
