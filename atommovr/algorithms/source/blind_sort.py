"""Blind rearrangement strategies for single-species atom arrays.

These algorithms generate a fixed move plan based solely on the array
geometry and target pattern, without using the actual occupancy data.
Every lattice site outside the target region receives a move toward
the target; sites that happen to be empty produce hardware no-ops
(the AOD activates, but no atom is present to transport).

Three strategies are implemented, differing in move ordering and
parallelism:

1. CompressAll  — simultaneous column compression then row compression.
2. ShellInward  — concentric rectangular shells, outermost first.
3. SweepLine    — edge-ordered sweep from all four boundaries inward.

All three target the MIDDLE_FILL configuration and produce axis-aligned
moves of exactly one lattice site per batch to minimise per-move
transport distance and the associated distance-dependent atom loss.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from atommovr.utils.Move import Move


def _target_bounds(target: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (row_min, row_max, col_min, col_max) of the target region."""
    indices = np.argwhere(target > 0)
    rows, cols = indices[:, 0], indices[:, 1]
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(val, hi))


#  Strategy 1: CompressAll


def _compress_all(state: np.ndarray, target: np.ndarray) -> List[List[Move]]:
    """Compress every site outside the target toward the target centre.

    Phase 1 (columns): for each column step, every site whose column index
    is outside the target column range receives a one-site horizontal move
    toward the nearest target column boundary.  All such moves within one
    step form a single parallel batch.

    Phase 2 (rows): analogous vertical compression.
    """
    rows, cols = state.shape
    r_min, r_max, c_min, c_max = _target_bounds(target)
    batches: List[List[Move]] = []

    # column compression — move sites left-of-target rightward, right-of-target leftward
    max_col_dist = max(c_min, cols - 1 - c_max)
    for _ in range(max_col_dist):
        batch: List[Move] = []
        for r in range(rows):
            for c in range(cols):
                if c < c_min:
                    batch.append(Move(r, c, r, c + 1))
                elif c > c_max:
                    batch.append(Move(r, c, r, c - 1))
        if batch:
            batches.append(batch)

    # row compression
    max_row_dist = max(r_min, rows - 1 - r_max)
    for _ in range(max_row_dist):
        batch = []
        for r in range(rows):
            for c in range(c_min, c_max + 1):
                if r < r_min:
                    batch.append(Move(r, c, r + 1, c))
                elif r > r_max:
                    batch.append(Move(r, c, r - 1, c))
        if batch:
            batches.append(batch)

    return batches


#  Strategy 2: ShellInward


def _shell_inward(state: np.ndarray, target: np.ndarray) -> List[List[Move]]:
    """Move atoms from concentric rectangular shells toward the target.

    Starting from the shell immediately adjacent to the target region,
    expand outward.  Each shell generates one column-compression batch
    (horizontal moves) followed by one row-compression batch (vertical
    moves), each moving atoms one site toward the target boundary.
    The inner shells move first so that destination sites are freed
    before the outer shells arrive.
    """
    rows, cols = state.shape
    r_min, r_max, c_min, c_max = _target_bounds(target)
    max_shells = max(r_min, rows - 1 - r_max, c_min, cols - 1 - c_max)
    batches: List[List[Move]] = []

    for shell in range(1, max_shells + 1):
        # rows spanned by this shell (clamp to grid)
        sr_lo = _clamp(r_min - shell, 0, rows - 1)
        sr_hi = _clamp(r_max + shell, 0, rows - 1)
        sc_lo = _clamp(c_min - shell, 0, cols - 1)
        sc_hi = _clamp(c_max + shell, 0, cols - 1)

        # horizontal moves: left and right columns of the shell
        h_batch: List[Move] = []
        for r in range(sr_lo, sr_hi + 1):
            if sc_lo < c_min:
                h_batch.append(Move(r, sc_lo, r, sc_lo + 1))
            if sc_hi > c_max:
                h_batch.append(Move(r, sc_hi, r, sc_hi - 1))
        if h_batch:
            batches.append(h_batch)

        # vertical moves: top and bottom rows of the shell
        v_batch: List[Move] = []
        for c in range(sc_lo, sc_hi + 1):
            if sr_lo < r_min:
                v_batch.append(Move(sr_lo, c, sr_lo + 1, c))
            if sr_hi > r_max:
                v_batch.append(Move(sr_hi, c, sr_hi - 1, c))
        if v_batch:
            batches.append(v_batch)

    return batches


#  Strategy 3: SweepLine


def _sweep_line(state: np.ndarray, target: np.ndarray) -> List[List[Move]]:
    """Edge-ordered sweep from all four boundaries toward the target.

    Each iteration simultaneously moves all four edges one step inward.
    Left-edge columns move right, right-edge columns move left, top rows
    move down, bottom rows move up.  This interleaves row and column
    moves within each global step, reducing the total number of batches
    compared to ShellInward when the target is not centred.
    """
    rows, cols = state.shape
    r_min, r_max, c_min, c_max = _target_bounds(target)

    left_cur, right_cur = 0, cols - 1
    top_cur, bot_cur = 0, rows - 1
    batches: List[List[Move]] = []

    while left_cur < c_min or right_cur > c_max or top_cur < r_min or bot_cur > r_max:
        # Build horizontal batch (left/right edge moves) separately
        h_batch: List[Move] = []
        if left_cur < c_min:
            for r in range(rows):
                h_batch.append(Move(r, left_cur, r, left_cur + 1))
            left_cur += 1
        if right_cur > c_max:
            for r in range(rows):
                h_batch.append(Move(r, right_cur, r, right_cur - 1))
            right_cur -= 1
        if h_batch:
            batches.append(h_batch)

        # Build vertical batch (top/bottom edge moves) separately
        v_batch: List[Move] = []
        if top_cur < r_min:
            for c in range(left_cur, right_cur + 1):
                v_batch.append(Move(top_cur, c, top_cur + 1, c))
            top_cur += 1
        if bot_cur > r_max:
            for c in range(left_cur, right_cur + 1):
                v_batch.append(Move(bot_cur, c, bot_cur - 1, c))
            bot_cur -= 1
        if v_batch:
            batches.append(v_batch)

    return batches


#  Dispatcher

STRATEGY_FUNCTIONS = {
    "compress": _compress_all,
    "shell": _shell_inward,
    "sweep": _sweep_line,
}


def blind_sort(
    state: np.ndarray,
    target: np.ndarray,
    strategy: str = "compress",
) -> tuple[np.ndarray, List[List[Move]], bool]:
    """Generate a blind rearrangement plan and simulate its effect on *state*.

    Parameters
    ----------
    state : 2-D occupancy array (may be ignored depending on strategy).
    target : 2-D binary target mask.
    strategy : one of ``"compress"``, ``"shell"``, ``"sweep"``.

    Returns
    -------
    final_state, move_batches, success_flag
    """
    if strategy not in STRATEGY_FUNCTIONS:
        raise ValueError(
            f"Unknown blind strategy '{strategy}'. "
            f"Choose from {list(STRATEGY_FUNCTIONS)}"
        )

    batches = STRATEGY_FUNCTIONS[strategy](state, target)

    # simulate the moves on a copy to compute the resulting state
    sim = state.copy()
    for batch in batches:
        clears, sets = [], []
        for mv in batch:
            if 0 <= mv.to_row < sim.shape[0] and 0 <= mv.to_col < sim.shape[1]:
                if (
                    sim[mv.from_row, mv.from_col] == 1
                    and sim[mv.to_row, mv.to_col] == 0
                ):
                    clears.append((mv.from_row, mv.from_col))
                    sets.append((mv.to_row, mv.to_col))
        for r, c in clears:
            sim[r, c] = 0
        for r, c in sets:
            sim[r, c] = 1

    # check success within the target region
    target_filled = np.sum(sim * target)
    success = int(target_filled) == int(np.sum(target))
    return sim, batches, success
