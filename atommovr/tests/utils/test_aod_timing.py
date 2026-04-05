"""
Authors: Claude, ChatGPT, Nikhil Harle
Description: Tests for AOD timing analysis functions
(see `atommovr.utils.aod_timing`)
"""

import pytest
import numpy as np
from numpy.typing import NDArray
from typing import List

from atommovr.utils.Move import Move

from atommovr.utils.aod_timing import (
    _get_pickup_accel_flags,  # tested
    _get_decel_putdown_flags,  # tested
    _detect_decel_and_putdown_masks,  # TODO add test confirming that unphysical/nonmatching moves don't get flagged
    _detect_pickup_and_accel_masks,  # TODO ^
    _remove_collisions_axis_np,  # tested
    _find_cross_axis_accel_tones,  # tested
    _classify_new_and_continuing_tones,  # tested
    _classify_fatal_and_nonfatal_colliding_tones,  # tested
)


class TestClassifyFatalAndNonfatalCollidingTones:
    def test_all_zeros(self):
        arr = np.zeros(5, dtype=np.uint8)
        arr_clean, fatal, nonfatal = _classify_fatal_and_nonfatal_colliding_tones(
            arr, return_clean=True
        )
        assert np.array_equal(arr_clean, arr)
        assert len(fatal) == 0
        assert len(nonfatal) == 0

    @pytest.mark.parametrize("val", [0, 1, 2, 3])
    def test_all_same(self, val):
        arr = np.ones(5, dtype=np.uint8) * val
        arr_clean, fatal, nonfatal = _classify_fatal_and_nonfatal_colliding_tones(
            arr, return_clean=True
        )
        assert np.array_equal(arr_clean, arr)
        assert len(fatal) == 0
        assert len(nonfatal) == 0

    @pytest.mark.parametrize(
        "config",
        [
            [3, 0, 2, 0],
            [2, 0, 0, 3],
            [2, 2, 0, 1],
            [1, 0, 3, 1],
            [3, 1, 2, 0],
            [3, 3, 2, 2],
        ],
    )
    def test_almost_configs(self, config):
        arr = np.asarray(config, dtype=np.uint8)
        arr_clean, fatal, nonfatal = _classify_fatal_and_nonfatal_colliding_tones(
            arr, return_clean=True
        )
        assert np.array_equal(arr_clean, arr)
        assert len(fatal) == 0
        assert len(nonfatal) == 0

    @pytest.mark.parametrize("x", [0, 1])
    def test_triad_x01(self, x):
        arr = np.asarray([0, 1, 2, x, 3, 0, 2], dtype=np.uint8)
        arr_clean, fatal, nonfatal = _classify_fatal_and_nonfatal_colliding_tones(
            arr, return_clean=True
        )
        assert np.array_equal([0, 1, 0, 0, 0, 0, 2], arr_clean)
        assert set(fatal) == {3}
        assert set(nonfatal) == {2, 4}

    def test_triad_x2(self):
        arr = np.asarray([0, 1, 2, 2, 3, 0, 2], dtype=np.uint8)
        arr_clean, fatal, nonfatal = _classify_fatal_and_nonfatal_colliding_tones(
            arr, return_clean=True
        )
        assert np.array_equal([0, 1, 0, 0, 0, 0, 2], arr_clean)
        assert set(fatal) == {3, 4}
        assert set(nonfatal) == {2}

    def test_triad_x3(self):
        arr = np.asarray([0, 1, 2, 3, 3, 0, 2], dtype=np.uint8)
        arr_clean, fatal, nonfatal = _classify_fatal_and_nonfatal_colliding_tones(
            arr, return_clean=True
        )
        assert np.array_equal([0, 1, 0, 0, 0, 0, 2], arr_clean)
        assert set(fatal) == {2, 3}
        assert set(nonfatal) == {4}

    def test_pair_21(self):
        arr = np.asarray([0, 1, 2, 1, 2, 0, 2], dtype=np.uint8)
        arr_clean, fatal, nonfatal = _classify_fatal_and_nonfatal_colliding_tones(
            arr, return_clean=True
        )
        assert np.array_equal([0, 1, 0, 0, 2, 0, 2], arr_clean)
        assert set(fatal) == {3}
        assert set(nonfatal) == {2}

    def test_pair_23(self):
        arr = np.asarray([0, 1, 2, 3, 2, 0, 2], dtype=np.uint8)
        arr_clean, fatal, nonfatal = _classify_fatal_and_nonfatal_colliding_tones(
            arr, return_clean=True
        )
        assert np.array_equal([0, 1, 0, 0, 2, 0, 2], arr_clean)
        assert set(fatal) == {2, 3}
        assert len(nonfatal) == 0

    def test_pair_13(self):
        arr = np.asarray([0, 1, 1, 3, 2, 0, 2], dtype=np.uint8)
        arr_clean, fatal, nonfatal = _classify_fatal_and_nonfatal_colliding_tones(
            arr, return_clean=True
        )
        assert np.array_equal([0, 1, 0, 0, 2, 0, 2], arr_clean)
        assert set(fatal) == {2}
        assert set(nonfatal) == {3}


class TestDetectPickupAndAccelMasksNew:
    def test_first_round_pickup_all_and_accel_if_moving(self) -> None:
        """
        First round special case: all moves require pickup; accel depends on whether
        the tone at the source is moving on either axis.
        """
        curr_h = np.array([0, 2, 0], dtype=np.int8)
        curr_v = np.array([0, 0, 1], dtype=np.int8)
        move_list = [Move(from_row=2, from_col=1, to_row=2, to_col=2)]

        pickup_mask, accel_mask = _detect_pickup_and_accel_masks(
            prev_h=None,
            prev_v=None,
            curr_h=curr_h,
            curr_v=curr_v,
            curr_move_set=move_list,
        )

        assert pickup_mask.dtype == np.bool_
        assert accel_mask.dtype == np.bool_
        assert pickup_mask.tolist() == [True]
        assert accel_mask.tolist() == [True]
        assert accel_mask.shape == (1,)
        assert pickup_mask.shape == (1,)

    def test_or_aggregation_propagates_to_move_sources(self) -> None:
        """
        If accel is required at a current tone index, any move whose source sits at
        that tone index should be accel-eligible.
        """
        prev_h = np.array([2, 1, 3], dtype=np.int8)
        curr_h = np.array([0, 2, 0], dtype=np.int8)  # accel at index 1 by OR rule
        prev_v = np.array([0, 1, 0], dtype=np.int8)
        curr_v = np.array([0, 1, 0], dtype=np.int8)

        move_list = [Move(from_row=1, from_col=1, to_row=1, to_col=2)]
        pickup_mask, accel_mask = _detect_pickup_and_accel_masks(
            prev_h=prev_h,
            prev_v=prev_v,
            curr_h=curr_h,
            curr_v=curr_v,
            curr_move_set=move_list,
        )
        assert pickup_mask.dtype == np.bool_
        assert accel_mask.dtype == np.bool_
        assert pickup_mask.tolist() == [False]
        assert accel_mask.tolist() == [True]
        assert accel_mask.shape == (1,)
        assert pickup_mask.shape == (1,)

    @pytest.mark.xfail(
        reason="detect_* functions do not yet check whether a move matches the AOD cmds",
        strict=True,
    )
    def test_doesnt_assign_events_to_unphysical_moves(self) -> None:
        prev_h = np.array([2, 1, 3], dtype=np.int8)
        curr_h = np.array([0, 2, 0], dtype=np.int8)  # accel at index 1 by OR rule
        prev_v = np.array([1, 0, 0], dtype=np.int8)
        curr_v = np.array([1, 0, 0], dtype=np.int8)

        move_list = [Move(from_row=0, from_col=1, to_row=0, to_col=0)]
        pickup_mask, accel_mask = _detect_pickup_and_accel_masks(
            prev_h=prev_h,
            prev_v=prev_v,
            curr_h=curr_h,
            curr_v=curr_v,
            curr_move_set=move_list,
        )
        assert pickup_mask.dtype == np.bool_
        assert accel_mask.dtype == np.bool_
        assert pickup_mask.tolist() == [False]
        assert accel_mask.tolist() == [False]
        assert accel_mask.shape == (1,)
        assert pickup_mask.shape == (1,)


class TestDetectDecelAndPutdownMasksNew:
    def test_handles_converging_tones(self) -> None:
        curr_h = np.array([0, 2, 1, 3], dtype=np.int8)
        next_h = np.array([0, 0, 1, 0], dtype=np.int8)
        curr_v = np.array([1, 0, 0], dtype=np.int8)
        next_v = np.array([1, 0, 0], dtype=np.int8)

        move_list = [
            Move(from_row=0, from_col=1, to_row=0, to_col=2),
            Move(0, 2, 0, 2),
            Move(0, 3, 0, 2),
        ]
        decel_mask, putdown_mask = _detect_decel_and_putdown_masks(
            curr_h=curr_h,
            curr_v=curr_v,
            next_h=next_h,
            next_v=next_v,
            curr_move_set=move_list,
        )
        assert putdown_mask.tolist() == [False, False, False]
        assert decel_mask.tolist() == [True, False, True]
        assert decel_mask.dtype == np.bool_
        assert putdown_mask.dtype == np.bool_
        assert decel_mask.shape == (3,)
        assert putdown_mask.shape == (3,)

    def test_mask_matches_move_order(self) -> None:
        curr_h = np.array([0, 2, 1, 3], dtype=np.int8)
        next_h = np.array([0, 0, 1, 0], dtype=np.int8)
        curr_v = np.array([1, 0, 0], dtype=np.int8)
        next_v = np.array([1, 0, 0], dtype=np.int8)

        move_list = [
            Move(0, 3, 0, 2),
            Move(from_row=0, from_col=1, to_row=0, to_col=2),
            Move(0, 2, 0, 2),
        ]
        decel_mask, putdown_mask = _detect_decel_and_putdown_masks(
            curr_h=curr_h,
            curr_v=curr_v,
            next_h=next_h,
            next_v=next_v,
            curr_move_set=move_list,
        )
        assert putdown_mask.tolist() == [False, False, False]
        assert decel_mask.tolist() == [True, True, False]
        assert decel_mask.dtype == np.bool_
        assert putdown_mask.dtype == np.bool_
        assert decel_mask.shape == (3,)
        assert putdown_mask.shape == (3,)

    def test_mask_matches_move_order2(self) -> None:
        curr_h = np.array([0, 1, 0, 3], dtype=np.int8)
        next_h = np.array([0, 1, 1, 0], dtype=np.int8)
        curr_v = np.array([1, 0, 3], dtype=np.int8)
        next_v = np.array([0, 1, 0], dtype=np.int8)

        move_list = [
            Move(0, 1, 0, 1),
            Move(0, 3, 0, 2),
            Move(2, 1, 1, 1),
            Move(2, 3, 1, 2),
        ]
        putdown_expected = [True, True, False, False]
        decel_expected = [False, True, True, True]

        decel_mask, putdown_mask = _detect_decel_and_putdown_masks(
            curr_h=curr_h,
            curr_v=curr_v,
            next_h=next_h,
            next_v=next_v,
            curr_move_set=move_list,
        )

        assert decel_mask.dtype == np.bool_
        assert putdown_mask.dtype == np.bool_
        assert decel_mask.shape == (4,)
        assert putdown_mask.shape == (4,)

        decel_mask, putdown_mask = _detect_decel_and_putdown_masks(
            curr_h=curr_h,
            curr_v=curr_v,
            next_h=next_h,
            next_v=next_v,
            curr_move_set=move_list[::-1],
        )
        assert putdown_mask.tolist() == putdown_expected[::-1]
        assert decel_mask.tolist() == decel_expected[::-1]

    @pytest.mark.xfail(
        reason="detect_* functions do not yet check whether a move matches the AOD cmds",
        strict=True,
    )
    def test_doesnt_assign_events_to_unphysical_moves(self) -> None:
        curr_h = np.array([0, 2, 0, 0], dtype=np.int8)
        next_h = np.array([0, 0, 1, 0], dtype=np.int8)
        curr_v = np.array([1, 0, 0], dtype=np.int8)
        next_v = np.array([1, 0, 0], dtype=np.int8)

        # incorrect move (doesn't correspond to AOD cmds)
        unphysical_move_list = [Move(from_row=1, from_col=1, to_row=1, to_col=2)]
        decel_mask, putdown_mask = _detect_decel_and_putdown_masks(
            curr_h=curr_h,
            curr_v=curr_v,
            next_h=next_h,
            next_v=next_v,
            curr_move_set=unphysical_move_list,
        )
        assert putdown_mask.tolist() == [False]
        assert decel_mask.tolist() == [False]
        assert decel_mask.dtype == np.bool_
        assert putdown_mask.dtype == np.bool_
        assert decel_mask.shape == (1,)
        assert putdown_mask.shape == (1,)


# TESTED/SANITY CHECKED


class TestDecelPutdownFlags:
    """
    Test single-axis pickup and accel flag extraction.
    Boolean string format:
    (needs_decel, needs_putdown)
    """

    def test_empty_inputs(self):
        curr = []
        nexx = []
        res = _get_decel_putdown_flags(curr, nexx)
        flags = res[:2]
        decel_inds, putdown_inds = res[2:]
        assert flags == (False, False)
        assert set(decel_inds) == set()
        assert set(putdown_inds) == set()

    def test_single_tone_continues(self) -> None:
        curr: List[int] = [0, 2, 0, 0, 0]
        nexx: List[int] = [0, 0, 2, 0, 0]
        res = _get_decel_putdown_flags(curr, nexx)
        flags = res[:2]
        decel_inds, putdown_inds = res[2:]
        assert flags == (False, False)
        assert set(decel_inds) == set()
        assert set(putdown_inds) == set()

    def test_single_tone_putdown(self) -> None:
        curr: List[int] = [0, 0, 0, 1, 0]
        nexx: List[int] = [0, 0, 0, 0, 0]
        res = _get_decel_putdown_flags(curr, nexx)
        flags = res[:2]
        decel_inds, putdown_inds = res[2:]
        assert flags == (False, True)
        assert set(decel_inds) == set()
        assert set(putdown_inds) == {3}

    def test_decel_fwd_and_putdown(self) -> None:
        curr: List[int] = [0, 0, 2, 0, 0]
        nexx: List[int] = [0, 0, 0, 0, 0]
        res = _get_decel_putdown_flags(curr, nexx)
        flags = res[:2]
        decel_inds, putdown_inds = res[2:]
        assert flags == (True, True)
        assert set(decel_inds) == {2}
        assert set(putdown_inds) == {2}

    def test_decel_bwd_and_putdown(self) -> None:
        curr: List[int] = [0, 3, 0, 0, 0]
        nexx: List[int] = [0, 0, 0, 0, 0]
        res = _get_decel_putdown_flags(curr, nexx)
        flags = res[:2]
        decel_inds, putdown_inds = res[2:]
        assert flags == (True, True)
        assert set(decel_inds) == {1}
        assert set(putdown_inds) == {1}

    def test_direction_change(self) -> None:
        curr: List[int] = [2, 0, 2, 0, 0]
        nexx: List[int] = [0, 3, 0, 3, 0]
        res = _get_decel_putdown_flags(curr, nexx)
        flags = res[:2]
        decel_inds, putdown_inds = res[2:]
        assert flags == (True, False)
        assert set(decel_inds) == {0, 2}
        assert set(putdown_inds) == set()

    def test_almost_direction_change(self) -> None:
        curr: List[int] = [0, 0, 2, 0, 0]
        nexx: List[int] = [0, 0, 0, 0, 3]
        res = _get_decel_putdown_flags(curr, nexx)
        flags = res[:2]
        decel_inds, putdown_inds = res[2:]
        assert flags == (True, True)
        assert set(decel_inds) == {2}
        assert set(putdown_inds) == {2}

    def test_decel_to_hold(self) -> None:
        curr: List[int] = [0, 0, 3, 0, 0]
        nexx: List[int] = [0, 1, 0, 0, 0]
        res = _get_decel_putdown_flags(curr, nexx)
        flags = res[:2]
        decel_inds, putdown_inds = res[2:]
        assert flags == (True, False)
        assert set(decel_inds) == {2}
        assert set(putdown_inds) == set()

    def test_eject_right(self) -> None:
        curr: List[int] = [1, 0, 0, 0, 2]
        nexx: List[int] = [0, 0, 1, 0, 0]
        res = _get_decel_putdown_flags(curr, nexx)
        flags = res[:2]
        decel_inds, putdown_inds = res[2:]
        assert flags == (False, True)
        assert set(decel_inds) == set()
        assert set(putdown_inds) == {0}

    def test_eject_left(self) -> None:
        curr: List[int] = [3, 0, 0, 0, 0]
        nexx: List[int] = [0, 1, 0, 0, 0]
        res = _get_decel_putdown_flags(curr, nexx)
        flags = res[:2]
        decel_inds, putdown_inds = res[2:]
        assert flags == (False, False)
        assert set(decel_inds) == set()
        assert set(putdown_inds) == set()

    def test_eject_right_multitone(self) -> None:
        curr: List[int] = [0, 3, 0, 0, 2]
        nexx: List[int] = [0, 0, 1, 0, 0]
        res = _get_decel_putdown_flags(curr, nexx)
        flags = res[:2]
        decel_inds, putdown_inds = res[2:]
        assert flags == (True, True)
        assert set(decel_inds) == {1}
        assert set(putdown_inds) == {1}


class TestPickupAccelFlags:
    """
    Test single-axis pickup and accel flag extraction.
    Boolean string format:
    (needs_pickup, needs_accel)
    """

    def test_empty_inputs(self):
        curr = []
        prev = []
        res = _get_pickup_accel_flags(prev, curr)
        flags = res[:2]
        pickup_inds, accel_inds = res[2:]
        assert flags == (False, False)
        assert set(pickup_inds) == set()
        assert set(accel_inds) == set()

    def test_prev_oob_accel(self):
        prev = [3, 0, 0]
        curr = [0, 0, 0]
        res = _get_pickup_accel_flags(prev, curr)
        flags = res[:2]
        pickup_inds, accel_inds = res[2:]
        assert flags == (False, False)
        assert set(pickup_inds) == set()
        assert set(accel_inds) == set()

    def test_single_tone_continues(self) -> None:
        prev: List[int] = [0, 2, 0, 0, 0]
        curr: List[int] = [0, 0, 2, 0, 0]
        res = _get_pickup_accel_flags(prev, curr)
        flags = res[:2]
        pickup_inds, accel_inds = res[2:]
        assert flags == (False, False)
        assert set(pickup_inds) == set()
        assert set(accel_inds) == set()

    def test_single_tone_pickup(self) -> None:
        prev: List[int] = [0, 0, 0, 0, 0]
        curr: List[int] = [0, 0, 0, 1, 0]
        res = _get_pickup_accel_flags(prev, curr)
        flags = res[:2]
        pickup_inds, accel_inds = res[2:]
        assert flags == (True, False)
        assert set(pickup_inds) == {3}
        assert set(accel_inds) == set()

    def test_multiple_pickups(self) -> None:
        prev: List[int] = [2, 0, 0, 1, 0, 0]
        curr: List[int] = [0, 0, 1, 0, 3, 0]
        res = _get_pickup_accel_flags(prev, curr)
        flags = res[:2]
        pickup_inds, accel_inds = res[2:]
        assert flags == (True, True)
        assert set(pickup_inds) == {2, 4}
        assert set(accel_inds) == {4}

    def test_pickup_and_accel_fwd(self) -> None:
        prev: List[int] = [0, 0, 0, 0, 0]
        curr: List[int] = [0, 0, 2, 0, 0]
        res = _get_pickup_accel_flags(prev, curr)
        flags = res[:2]
        pickup_inds, accel_inds = res[2:]
        assert flags == (True, True)
        assert set(pickup_inds) == {2}
        assert set(accel_inds) == {2}

    def test_pickup_and_accel_bwd(self) -> None:
        prev: List[int] = [0, 0, 0, 0, 0]
        curr: List[int] = [0, 3, 0, 0, 0]
        res = _get_pickup_accel_flags(prev, curr)
        flags = res[:2]
        pickup_inds, accel_inds = res[2:]
        assert flags == (True, True)
        assert set(pickup_inds) == {1}
        assert set(accel_inds) == {1}

    def test_direction_change(self) -> None:
        prev: List[int] = [0, 2, 0, 0, 0]
        curr: List[int] = [0, 0, 3, 0, 0]
        res = _get_pickup_accel_flags(prev, curr)
        flags = res[:2]
        pickup_inds, accel_inds = res[2:]
        assert flags == (False, True)
        assert set(pickup_inds) == set()
        assert set(accel_inds) == {2}

    def test_almost_direction_change(self) -> None:
        prev: List[int] = [0, 2, 0, 0, 0]
        curr: List[int] = [0, 0, 0, 0, 3]
        res = _get_pickup_accel_flags(prev, curr)
        flags = res[:2]
        pickup_inds, accel_inds = res[2:]
        assert flags == (True, True)
        assert set(pickup_inds) == {4}
        assert set(accel_inds) == {4}

    def test_decel_to_hold(self) -> None:
        prev: List[int] = [0, 2, 0, 0, 0]
        curr: List[int] = [0, 0, 1, 0, 0]
        res = _get_pickup_accel_flags(prev, curr)
        flags = res[:2]
        pickup_inds, accel_inds = res[2:]
        assert flags == (False, False)
        assert set(pickup_inds) == set()
        assert set(accel_inds) == set()

    def test_eject_right(self) -> None:
        prev: List[int] = [1, 0, 0, 0, 0]
        curr: List[int] = [0, 0, 0, 0, 2]
        res = _get_pickup_accel_flags(prev, curr)
        flags = res[:2]
        pickup_inds, accel_inds = res[2:]
        assert flags == (True, True)
        assert set(pickup_inds) == {4}
        assert set(accel_inds) == {4}

    def test_eject_left(self) -> None:
        prev: List[int] = [0, 0, 1, 0, 0]
        curr: List[int] = [3, 0, 0, 0, 0]
        res = _get_pickup_accel_flags(prev, curr)
        flags = res[:2]
        pickup_inds, accel_inds = res[2:]
        assert flags == (True, True)
        assert set(pickup_inds) == {0}
        assert set(accel_inds) == {0}

    def test_pickup_accel_prev_all_zero_curr_all_zero(self) -> None:
        prev = [0, 0, 0]
        curr = [0, 0, 0]
        needs_pickup, needs_accel, pickup_inds, accel_inds = _get_pickup_accel_flags(
            prev, curr
        )
        assert (needs_pickup, needs_accel) == (False, False)
        assert set(pickup_inds) == set()
        assert set(accel_inds) == set()

    def test_or_aggregation_for_converging_preimages(self) -> None:
        """
        When multiple predecessor tones map to the same current index, accel should
        be the OR over the per-preimage accel requirements.
        """
        prev_cmds = [2, 1, 3]

        # x=1 -> no accel
        needs_p, needs_a, p_inds, a_inds = _get_pickup_accel_flags(prev_cmds, [0, 1, 0])
        assert needs_p is False
        assert needs_a is False
        assert np.array_equal(p_inds, [])
        assert np.array_equal(a_inds, [])

        # x=2 -> accel required at index 1 (due to some preimages needing accel)
        needs_p, needs_a, p_inds, a_inds = _get_pickup_accel_flags(prev_cmds, [0, 2, 0])
        assert needs_p is False
        assert needs_a is True
        assert np.array_equal(p_inds, [])
        assert np.array_equal(a_inds, [1])

        # x=3 -> accel required at index 1
        needs_p, needs_a, p_inds, a_inds = _get_pickup_accel_flags(prev_cmds, [0, 3, 0])
        assert needs_p is False
        assert needs_a is True
        assert np.array_equal(p_inds, [])
        assert np.array_equal(a_inds, [1])

    def test_new_tone_without_preimage_triggers_pickup_and_accel(self) -> None:
        """
        A current nonzero tone with no predecessor mapping is treated as a
        "single-tone remaining current" (prev=0) and can require pickup/accel.
        """
        prev_cmds = [0, 0, 0]
        curr_cmds = [0, 2, 0]
        needs_p, needs_a, p_inds, a_inds = _get_pickup_accel_flags(prev_cmds, curr_cmds)

        assert needs_p is True
        assert needs_a is True
        assert np.array_equal(p_inds, [1])
        assert np.array_equal(a_inds, [1])

    def test_collision_patterns_do_not_require_cleaned_tones(self) -> None:
        """
        Regression: function should not require collision-cleaned inputs and should
        behave deterministically under converging patterns like [2,1].
        """
        prev_cmds = [2, 1]
        curr_cmds = [0, 1]
        needs_p, needs_a, p_inds, a_inds = _get_pickup_accel_flags(prev_cmds, curr_cmds)

        assert needs_p is False
        assert needs_a is False
        assert np.array_equal(p_inds, [])
        assert np.array_equal(a_inds, [])


class TestClassifyTonesAsNewOrExisting:

    def test_classify_static_continuation(self) -> None:
        prev = [0, 1, 0]
        curr = [0, 1, 0]
        new, cont = _classify_new_and_continuing_tones(prev, curr)
        assert np.array_equal(new, [])
        assert np.array_equal(cont, [1])

    def test_classify_simple_new_static_tone(self) -> None:
        prev = [0, 0, 0]
        curr = [0, 1, 0]
        new, cont = _classify_new_and_continuing_tones(prev, curr)
        assert np.array_equal(new, [1])
        assert np.array_equal(cont, [])

    def test_classify_continuing_motion(self) -> None:
        prev = [0, 2, 0, 0]
        curr = [0, 0, 2, 0]
        new, cont = _classify_new_and_continuing_tones(prev, curr)
        assert np.array_equal(new, [])
        assert np.array_equal(cont, [2])

    def test_classify_continuing_motion_change_direction(self) -> None:
        prev = [0, 2, 0, 0]
        curr = [0, 0, 3, 0]
        new, cont = _classify_new_and_continuing_tones(prev, curr)
        assert np.array_equal(new, [])
        assert np.array_equal(cont, [2])

    def test_classify_decelerating_tweezer(self) -> None:
        prev = [0, 2, 0, 0]
        curr = [0, 0, 1, 0]
        new, cont = _classify_new_and_continuing_tones(prev, curr)
        assert np.array_equal(new, [])
        assert np.array_equal(cont, [2])

    def test_classify_new_moving_tone(self) -> None:
        prev = [0, 0, 0]
        curr = [0, 2, 0]
        new, cont = _classify_new_and_continuing_tones(prev, curr)
        assert np.array_equal(new, [1])
        assert np.array_equal(cont, [])

    def test_classify_mixed_new_and_continuing_motion(self) -> None:
        prev = [0, 2, 0, 0]
        curr = [1, 0, 2, 0]
        new, cont = _classify_new_and_continuing_tones(prev, curr)
        assert np.array_equal(new, [0])
        assert np.array_equal(cont, [2])

    def test_classify_boundary_motion_no_index_error(self) -> None:
        # Tone at last index "moving right" would go OOB; curr has no tone.
        prev = [0, 0, 0, 2]
        curr = [0, 0, 0, 0]
        new, cont = _classify_new_and_continuing_tones(prev, curr)
        assert np.array_equal(new, [])
        assert np.array_equal(cont, [])

    def test_classify_same_index_command_change_is_continuing(self) -> None:
        prev = [0, 1, 0]
        curr = [0, 2, 0]
        new, cont = _classify_new_and_continuing_tones(prev, curr)
        assert np.array_equal(new, [])
        assert np.array_equal(cont, [1])


class TestCollisionRemoval:
    """Test collision detection and removal."""

    def test_no_collision(self) -> None:
        input_arr: NDArray[np.int_] = np.array([0, 2, 0, 0, 0])
        result: NDArray[np.int_] = _remove_collisions_axis_np(input_arr)
        expected: NDArray[np.int_] = np.array([0, 2, 0, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_collision_2_1(self) -> None:
        input_arr: NDArray[np.int_] = np.array([0, 2, 1, 0, 0])
        result: NDArray[np.int_] = _remove_collisions_axis_np(input_arr)
        expected: NDArray[np.int_] = np.array([0, 0, 0, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_collision_1_3(self) -> None:
        input_arr: NDArray[np.int_] = np.array([0, 1, 3, 0, 1])
        result: NDArray[np.int_] = _remove_collisions_axis_np(input_arr)
        expected: NDArray[np.int_] = np.array([0, 0, 0, 0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_collision_2_3(self) -> None:
        input_arr: NDArray[np.int_] = np.array([0, 2, 3, 1, 0])
        result: NDArray[np.int_] = _remove_collisions_axis_np(input_arr)
        expected: NDArray[np.int_] = np.array([0, 0, 0, 1, 0])
        np.testing.assert_array_equal(result, expected)

    def test_collision_2_x_3(self) -> None:
        input_arr1: NDArray[np.int_] = np.array([2, 0, 3, 0, 1])
        result1: NDArray[np.int_] = _remove_collisions_axis_np(input_arr1)
        expected1: NDArray[np.int_] = np.array([0, 0, 0, 0, 1])
        np.testing.assert_array_equal(result1, expected1)

        input_arr2: NDArray[np.int_] = np.array([2, 1, 3, 1, 0])
        result2: NDArray[np.int_] = _remove_collisions_axis_np(input_arr2)
        expected2: NDArray[np.int_] = np.array([0, 0, 0, 1, 0])
        np.testing.assert_array_equal(result2, expected2)

    def test_multiple_independent_tones(self) -> None:
        input_arr: NDArray[np.int_] = np.array([2, 0, 0, 3, 1])
        result: NDArray[np.int_] = _remove_collisions_axis_np(input_arr)
        expected: NDArray[np.int_] = np.array([2, 0, 0, 3, 1])
        np.testing.assert_array_equal(result, expected)


class TestCrossAxisAccelTones:

    def test_cross_axis_no_new_tones(self) -> None:
        prev_x = [0, 0, 0, 1, 0]
        prev_y = [0, 2, 0, 0, 0]

        curr_x = [0, 0, 0, 1, 0]
        curr_y = [0, 0, 2, 0, 0]  # continuation of motion

        ax, ay = _find_cross_axis_accel_tones(prev_x, curr_x, prev_y, curr_y)

        assert np.array_equal(ax, [])
        assert np.array_equal(ay, [])

    def test_cross_axis_tone_jump(self) -> None:
        prev_x = [0, 0, 0, 1, 0]
        prev_y = [0, 2, 0, 0, 0]

        curr_x = [1, 0, 0, 0, 0]
        curr_y = [0, 0, 2, 0, 0]

        ax, ay = _find_cross_axis_accel_tones(prev_x, curr_x, prev_y, curr_y)

        assert np.array_equal(ax, [0])
        assert np.array_equal(ay, [])

    def test_cross_axis_new_intersects_moving(self) -> None:
        prev_x = [0, 0, 0, 1, 0]
        prev_y = [0, 2, 0, 0, 0]

        curr_x = [1, 0, 0, 1, 0]  # new static tone
        curr_y = [0, 0, 2, 0, 0]  # moving tone continues

        ax, ay = _find_cross_axis_accel_tones(prev_x, curr_x, prev_y, curr_y)

        assert np.array_equal(ax, [0])
        assert np.array_equal(ay, [])

    def test_cross_axis_multiple_new_tones(self) -> None:
        prev_x = [0, 0, 0, 1, 0]
        prev_y = [0, 2, 0, 0, 0]

        curr_x = [1, 0, 0, 1, 1]  # new static tone
        curr_y = [0, 0, 2, 0, 0]  # moving tone continues

        ax, ay = _find_cross_axis_accel_tones(prev_x, curr_x, prev_y, curr_y)

        assert np.array_equal(ay, [])
        assert set(ax) == {0, 4}

    def test_cross_axis_new_static_new_moving(self) -> None:
        prev_x = [0, 2, 0, 0, 0]
        prev_y = [0, 0, 0, 1, 0]

        curr_x = [0, 0, 2, 0, 0]  # moving tone continues
        curr_y = [1, 0, 0, 1, 2]  # new static tone

        ax, ay = _find_cross_axis_accel_tones(prev_x, curr_x, prev_y, curr_y)

        assert np.array_equal(ax, [])
        assert set(ay) == {0}

    def test_cross_axis_new_without_accel(self) -> None:
        prev_x = [0, 0, 0, 1, 0]
        prev_y = [0, 2, 0, 0, 0]

        curr_x = [0, 0, 0, 1, 0]
        curr_y = [1, 0, 2, 0, 0]  # new static tone, but no moving x

        ax, ay = _find_cross_axis_accel_tones(prev_x, curr_x, prev_y, curr_y)

        assert np.array_equal(ax, [])
        assert np.array_equal(ay, [])

    def test_cross_axis_multiple_new_only_one_accel(self) -> None:
        prev_x = [0, 0, 0, 1, 0]
        prev_y = [0, 2, 0, 0, 0]

        curr_x = [1, 0, 0, 1, 0]
        curr_y = [1, 0, 2, 0, 0]

        ax, ay = _find_cross_axis_accel_tones(prev_x, curr_x, prev_y, curr_y)

        assert np.array_equal(ax, [0])
        assert np.array_equal(ay, [])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
