"""
Integration seam tests for the move-event pipeline.

These tests verify compatibility across the following modules:

- `atommovr.utils.aod_timing`
  (detects per-move pickup/accel/decel/putdown eligibility from AOD command timing)
- `atommovr.utils.ErrorModel`
  (applies deterministic/stochastic failure bits into a per-move event mask)
- `atommovr.utils.error_utils`
  (finalizes event masks: suppression, primary-event resolution, and write-back to Move objects)
- `atommovr.utils.failure_policy`
  (bit semantics, suppression rules, and primary-event precedence)
- `atommovr.utils.Move`
  (stores per-move failure event / failure flag)

Scope
-----
These tests intentionally stop short of `AtomArray.move_atoms()` and instead exercise the
“integration seam” between low-level AOD tone analysis and error/failure resolution. This keeps
failures local and easier to debug while still testing the end-to-end event pipeline.
"""

import numpy as np

from atommovr.utils.Move import Move
from atommovr.utils.ErrorModel import ErrorModel
from atommovr.utils.aod_timing import (
    _detect_pickup_and_accel_masks,
    _detect_decel_and_putdown_masks,
)
from atommovr.utils.error_utils import finalize_events_to_moves
from atommovr.utils.failure_policy import (
    FailureBit,
    FailureEvent,
    FailureFlag,
    bit_value,
)


def _mask_of(*bits: FailureBit) -> np.uint64:
    m = np.uint64(0)
    for b in bits:
        m |= bit_value(b)
    return m


def test_REGRESSION_pipeline_seam_converging_next_tones_do_not_force_decel_putdown() -> (
    None
):
    """
    Regression seam test: a converging/crossing pattern in *next* commands must not
    create a spurious decel/putdown eligibility on the current move.

    Motivation:
        Older logic that "cleaned" next-tone patterns (e.g. [2,3]) could erase
        real next-step tones, incorrectly making it look like the current tone
        disappears, which would then (wrongly) trigger decel/putdown.
    """
    moves = [Move(1, 1, 1, 2)]
    event_mask = np.zeros(1, dtype=np.uint64)

    # Current round: horizontal source tone at col=1 is moving right (2).
    # Vertical axis is stationary-hold at row=1.
    curr_h_clean = np.array([0, 2, 0, 0], dtype=np.int8)
    curr_v_clean = np.array([0, 1, 0], dtype=np.int8)

    # Next round: the intended continuation of the moving tone lives at col=2 (2),
    # but there is also a neighboring left-moving tone at col=3 (3), forming [2,3].
    # This pattern would have been "cleaned away" in older code paths.
    next_h_clean = np.array([0, 0, 2, 3], dtype=np.int8)
    next_v_clean = np.array([0, 1, 0], dtype=np.int8)

    decel_mask, putdown_mask = _detect_decel_and_putdown_masks(
        curr_h=curr_h_clean,
        curr_v=curr_v_clean,
        next_h=next_h_clean,
        next_v=next_v_clean,
        curr_move_set=moves,
    )

    assert decel_mask.tolist() == [False]
    assert putdown_mask.tolist() == [False]

    # Even with fail rates = 1, no eligibility => no bits tagged => SUCCESS.
    em = ErrorModel(
        decel_fail_rate=1.0,
        putdown_fail_rate=1.0,
        seed=0,
    )
    em.apply_decel_errors_mask(event_mask, decel_mask)
    em.apply_putdown_errors_mask(event_mask, putdown_mask)

    primary = finalize_events_to_moves(moves, event_mask, store_mask_on_move=True)
    assert primary.tolist() == [int(FailureEvent.SUCCESS)]
    assert moves[0].fail_event == FailureEvent.SUCCESS
    assert moves[0].fail_flag == FailureFlag.SUCCESS


def test_aod_timing_pickup_accel_or_policy_converging_preimages_accel_triggers() -> (
    None
):
    """
    Regression/integration seam test for the “OR over predecessor tones” rule.

    Scenario (1D along horizontal axis):
      prev_h = [2, 1, 3]  -> all three predecessor tones map into the same destination index (1)
      curr_h = [0, 2, 0]  -> destination tone is moving (2)

    The OR-aggregation rule says: if any predecessor that maps into the destination would
    require accel, then the destination tone should be treated as accel-eligible.

    We keep vertical axis static to ensure the accel decision is purely horizontal here.
    """
    moves = [Move(0, 1, 0, 2)]  # source col = 1, consistent with the destination index
    prev_h = np.array([2, 1, 3], dtype=np.int8)
    prev_v = np.array([1], dtype=np.int8)
    curr_h = np.array([0, 2, 0], dtype=np.int8)
    curr_v = np.array([1], dtype=np.int8)

    pickup_mask, accel_mask = _detect_pickup_and_accel_masks(
        prev_h=prev_h,
        prev_v=prev_v,
        curr_h=curr_h,
        curr_v=curr_v,
        curr_move_set=moves,
    )

    # In this converging-preimage case, pickup should be False (tone existed via predecessors),
    # but accel should be True by the OR rule when the destination is a moving tone (2).
    assert pickup_mask.tolist() == [False]
    assert accel_mask.tolist() == [True]

    # Also verify the pipeline seam: accel-only tagging resolves to ACCEL_FAIL under p=1.
    event_mask = np.zeros(1, dtype=np.uint64)
    em = ErrorModel(accel_fail_rate=1.0, seed=0)
    em.apply_accel_errors_mask(event_mask, accel_mask)
    primary = finalize_events_to_moves(moves, event_mask, store_mask_on_move=True)

    assert primary.tolist() == [int(FailureEvent.ACCEL_FAIL)]
    assert moves[0].fail_event == FailureEvent.ACCEL_FAIL
    assert moves[0].fail_flag == FailureFlag.LOSS
    assert moves[0].fail_mask == int(_mask_of(FailureBit.ACCEL_FAIL))


def test_aod_timing_decel_putdown_not_triggered_by_converging_next_tones_elsewhere() -> (
    None
):
    """
    Regression/integration seam test for the bug you described:

    If a moving tone continues moving into the next step, we should NOT mark decel/putdown
    at the end of the current step, even if the *next-step* command list contains a
    converging-tone pattern elsewhere.

    Concrete scenario:
      curr_v: moving tone at index 0 (2) -> destination index 1
      next_v: moving tone continues at index 1 (2) AND there is also a converging tone nearby
              (the presence of both 2 and 3 in next_v should not force decel/putdown here)

    We keep horizontal tones static.
    """
    moves = [Move(0, 0, 1, 0)]  # vertical move down by 1 (tone 2 at v index 0)
    curr_h = np.array([1, 0, 0, 0], dtype=np.int8)
    curr_v = np.array([2, 0, 0, 0], dtype=np.int8)

    next_h = np.array([1, 0, 0, 0], dtype=np.int8)
    next_v = np.array([0, 2, 3, 0], dtype=np.int8)

    decel_mask, putdown_mask = _detect_decel_and_putdown_masks(
        curr_h=curr_h,
        curr_v=curr_v,
        next_h=next_h,
        next_v=next_v,
        curr_move_set=moves,
    )

    assert decel_mask.tolist() == [False]
    assert putdown_mask.tolist() == [False]

    # Seam check: no bits -> SUCCESS
    event_mask = np.zeros(1, dtype=np.uint64)
    em = ErrorModel(decel_fail_rate=1.0, putdown_fail_rate=1.0, seed=0)
    em.apply_decel_errors_mask(event_mask, decel_mask)
    em.apply_putdown_errors_mask(event_mask, putdown_mask)
    primary = finalize_events_to_moves(moves, event_mask, store_mask_on_move=True)

    assert primary.tolist() == [int(FailureEvent.SUCCESS)]
    assert moves[0].fail_event == FailureEvent.SUCCESS
    assert moves[0].fail_flag == FailureFlag.SUCCESS
    assert moves[0].fail_mask == 0


def test_aod_timing_event_pipeline_seam_first_round_pickup_and_accel() -> None:
    """
    First-round special case in aod_timing:
    - all current moves require pickup
    - accel depends on whether the source tone is moving on either axis
    Then verify ErrorModel tags + finalization write back correctly.
    """
    # Move source is (row=1, col=1)
    moves = [Move(1, 1, 1, 2)]
    event_mask = np.zeros(1, dtype=np.uint64)

    # First round => prev_*_clean = None
    # Current horizontal tone at source col=1 is moving (2), vertical is hold (1).
    curr_h_clean = np.array([0, 2, 0, 0], dtype=np.int8)
    curr_v_clean = np.array([0, 1, 0], dtype=np.int8)

    pickup_mask, accel_mask = _detect_pickup_and_accel_masks(
        prev_h=None,
        prev_v=None,
        curr_h=curr_h_clean,
        curr_v=curr_v_clean,
        curr_move_set=moves,
    )

    assert pickup_mask.tolist() == [True]
    assert accel_mask.tolist() == [True]

    em = ErrorModel(
        pickup_fail_rate=1.0,
        accel_fail_rate=1.0,
        seed=0,
    )
    em.apply_pickup_errors_mask(event_mask, pickup_mask)
    em.apply_accel_errors_mask(event_mask, accel_mask)

    primary = finalize_events_to_moves(moves, event_mask, store_mask_on_move=True)

    # pickup suppresses accel in policy
    assert primary.tolist() == [int(FailureEvent.PICKUP_FAIL)]
    assert moves[0].fail_event == FailureEvent.PICKUP_FAIL
    assert moves[0].fail_flag == FailureFlag.NO_PICKUP
    assert moves[0].fail_mask == int(_mask_of(FailureBit.PICKUP_FAIL))


def test_aod_timing_event_pipeline_seam_end_round_decel_and_putdown() -> None:
    """
    Use a simple one-move scenario where aod_timing marks decel+putdown,
    then verify policy suppression resolves to DECEL_FAIL when both bits are tagged.
    """
    # Source (row=1, col=1), horizontal tone moves right this round.
    moves = [Move(1, 1, 1, 2)]
    event_mask = np.zeros(1, dtype=np.uint64)

    curr_h_clean = np.array(
        [0, 2, 0, 0], dtype=np.int8
    )  # moving tone at col 1 (dest index 2)
    curr_v_clean = np.array([0, 1, 0], dtype=np.int8)  # hold tone at row 1

    # Next round: horizontal destination tone disappears -> decel + putdown at source tone
    next_h_clean = np.array([0, 0, 0, 0], dtype=np.int8)
    next_v_clean = np.array([0, 1, 0], dtype=np.int8)

    decel_mask, putdown_mask = _detect_decel_and_putdown_masks(
        curr_h=curr_h_clean,
        curr_v=curr_v_clean,
        next_h=next_h_clean,
        next_v=next_v_clean,
        curr_move_set=moves,
    )

    assert decel_mask.tolist() == [True]
    assert putdown_mask.tolist() == [True]

    em = ErrorModel(
        decel_fail_rate=1.0,
        putdown_fail_rate=1.0,
        seed=0,
    )
    em.apply_decel_errors_mask(event_mask, decel_mask)
    em.apply_putdown_errors_mask(event_mask, putdown_mask)

    primary = finalize_events_to_moves(moves, event_mask, store_mask_on_move=True)

    # In current suppression policy, decel suppresses putdown.
    assert primary.tolist() == [int(FailureEvent.DECEL_FAIL)]
    assert moves[0].fail_event == FailureEvent.DECEL_FAIL
    assert moves[0].fail_flag == FailureFlag.LOSS
    assert moves[0].fail_mask == int(_mask_of(FailureBit.DECEL_FAIL))


def test_aod_timing_event_pipeline_seam_empty_move_list_noop() -> None:
    """
    Explicit no-op seam test:
    empty move list + empty command lists should remain a no-op across
    aod_timing detection, ErrorModel application, and event finalization.
    """
    moves: list[Move] = []
    event_mask = np.zeros(0, dtype=np.uint64)

    pickup_mask, accel_mask = _detect_pickup_and_accel_masks(
        prev_h=None,
        prev_v=None,
        curr_h=np.array([], dtype=np.int8),
        curr_v=np.array([], dtype=np.int8),
        curr_move_set=moves,
    )
    decel_mask, putdown_mask = _detect_decel_and_putdown_masks(
        curr_h=np.array([], dtype=np.int8),
        curr_v=np.array([], dtype=np.int8),
        next_h=np.array([], dtype=np.int8),
        next_v=np.array([], dtype=np.int8),
        curr_move_set=moves,
    )

    assert pickup_mask.shape == (0,)
    assert accel_mask.shape == (0,)
    assert decel_mask.shape == (0,)
    assert putdown_mask.shape == (0,)

    em = ErrorModel(
        pickup_fail_rate=1.0,
        accel_fail_rate=1.0,
        decel_fail_rate=1.0,
        putdown_fail_rate=1.0,
        seed=0,
    )

    em.apply_pickup_errors_mask(event_mask, pickup_mask)
    em.apply_accel_errors_mask(event_mask, accel_mask)
    em.apply_decel_errors_mask(event_mask, decel_mask)
    em.apply_putdown_errors_mask(event_mask, putdown_mask)

    primary = finalize_events_to_moves(moves, event_mask, store_mask_on_move=True)
    assert primary.shape == (0,)
    assert primary.dtype == np.int32
