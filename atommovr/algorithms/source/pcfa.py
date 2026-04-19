"""Parallel Compression Filling Algorithm (PCFA).

The implementation follows the three-stage protocol described in the PCFA paper [1]_ and
enforces collision-free row motion by construction:

1. Initial validation: ensure the array contains at least as many atoms as the target mask.
2. Row compression: process columns sequentially, moving the k-th atom of every target row
        onto the k-th column of the left-aligned target window. This ordering guarantees that
        no atom overtakes another, and the batches respect the requested degree of parallelism (DOP).
3. Gap filling: identify deficit rows, prioritize them by the number of missing atoms,
        and borrow atoms from rows with the largest surplus outside the target window. Donor
        selection prefers straight-line moves (same column or same row) and falls back to
        shortest two-step routes when necessary, again limited by the DOP parameter.

The planner returns ``(final_state, move_batches, success_flag)`` where ``success_flag``
indicates whether the target mask is completely filled after executing the generated
move batches.

References
----------
.. [1] Y. Zhang et al., "A Fast Rearrangement Method for Defect-Free Atom Arrays,"
       Photonics 12(2), 117 (2025). https://doi.org/10.3390/photonics12020117
"""

from __future__ import annotations

# Copyright (c) 2026 Sascha Benz
# Ludwig-Maximilians-Universitaet Muenchen (Ryd-Yb tweezer array group)
# SPDX-License-Identifier: MIT
#
# Implementation author: Sascha Benz
# This file provides an independent reimplementation of the PCFA algorithm
# described in: Zhang, Y., Zhang, Z., Zhang, G., Zhang, Z., Chen, Y., Li, Y., Liu, W., Wu, J.,
# Sovkov, V., & Ma, J. (2025). A Fast Rearrangement Method for Defect-Free Atom Arrays.
# Photonics, 12(2), 117. https://doi.org/10.3390/photonics12020117
#

from typing import Dict, List, Sequence, Tuple

import numpy as np

from atommovr.utils.Move import Move


def _target_bounds_from_mask(target_mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (row_min, row_max_exclusive, col_min, col_max_exclusive) for targets."""
    if target_mask.ndim != 2:
        raise ValueError("Target mask must be 2D for PCFA.")
    indices = np.argwhere(target_mask > 0)
    if indices.size == 0:
        return 0, 0, 0, 0
    rows = indices[:, 0]
    cols = indices[:, 1]
    return (
        int(rows.min()),
        int(rows.max()) + 1,
        int(cols.min()),
        int(cols.max()) + 1,
    )


def _chunk_batch(batch: Sequence[Move], dop: int | None) -> List[List[Move]]:
    if not batch:
        return []
    if dop is None or dop <= 0:
        return [list(batch)]
    chunked: List[List[Move]] = []
    for idx in range(0, len(batch), dop):
        chunked.append(list(batch[idx : idx + dop]))
    return chunked


def _apply_batches_to_state(
    state: np.ndarray, batches: Sequence[Sequence[Move]]
) -> None:
    for batch in batches:
        clears = []
        sets = []
        for mv in batch:
            if not (
                0 <= mv.to_row < state.shape[0] and 0 <= mv.to_col < state.shape[1]
            ):
                continue
            if state[mv.from_row, mv.from_col] == 1:
                clears.append((mv.from_row, mv.from_col))
                sets.append((mv.to_row, mv.to_col))

        for r, c in clears:
            state[r, c] = 0
        for r, c in sets:
            state[r, c] = 1


def _row_path_clear(state: np.ndarray, row: int, src: int, dst: int) -> bool:
    if src == dst:
        return True
    step = 1 if dst > src else -1
    for col in range(src + step, dst, step):
        if state[row, col] == 1:
            return False
    return True


def _column_path_clear(state: np.ndarray, col: int, src: int, dst: int) -> bool:
    if src == dst:
        return True
    step = 1 if dst > src else -1
    for row in range(src + step, dst, step):
        if state[row, col] == 1:
            return False
    return True


def _plan_row_compression(
    state: np.ndarray,
    row_min: int,
    row_max: int,
    col_min: int,
    width: int,
    dop: int | None,
) -> List[List[Move]]:
    """Stage 2: sequential row-by-row compression.

    Moves ALL atoms in each target row to be contiguous starting at col_min.
    This ensures the leftmost atom goes to the leftmost target site, and surplus
    atoms are compacted to the right of the target region to facilitate gap filling.
    """
    compression_batches: List[List[Move]] = []

    # Iterate over each row in the target vertical range
    for r in range(row_min, row_max):
        # Get all atoms in the row
        atom_cols = sorted(np.where(state[r, :] == 1)[0])

        if not atom_cols:
            continue

        # We want to compact all atoms to the left, starting at col_min.
        # Mapping: atom_cols[i] -> col_min + i

        # Check if already compacted
        is_compacted = True
        for i, src in enumerate(atom_cols):
            dst = col_min + i
            if src != dst:
                is_compacted = False
                break

        if is_compacted:
            continue

        row_moves: List[Move] = []
        for i, src in enumerate(atom_cols):
            dst = col_min + i

            # Ensure we don't move off the grid
            # if dst >= state.shape[1]:
            # 	break

            if src == dst:
                continue

            row_moves.append(Move(r, src, r, dst))

        if not row_moves:
            continue

        # Sort moves to allow safe sequential execution if chunked.
        # Left movers (src > dst): Sort by dst Ascending.
        # Right movers (src < dst): Sort by dst Descending.
        left_movers = [m for m in row_moves if m.from_col > m.to_col]
        right_movers = [m for m in row_moves if m.from_col < m.to_col]

        left_movers.sort(key=lambda m: m.to_col)
        right_movers.sort(key=lambda m: m.to_col, reverse=True)

        # Combine: Left movers first, then Right movers.
        sorted_moves = left_movers + right_movers

        # Chunk by DOP
        chunks = _chunk_batch(sorted_moves, dop)
        compression_batches.extend(chunks)
        _apply_batches_to_state(state, chunks)

    return compression_batches


def _collect_donors_by_row(
    state: np.ndarray,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
) -> Dict[int, List[int]]:
    donors: Dict[int, List[int]] = {}
    for r in range(state.shape[0]):
        cols = np.where(state[r, :] == 1)[0]
        donor_cols = [
            int(c)
            for c in cols
            if not (row_min <= r < row_max and col_min <= c < col_max)
        ]
        if donor_cols:
            donors[r] = donor_cols
    return donors


def _ordered_vacancies(
    state: np.ndarray,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
) -> List[Tuple[int, int]]:
    vacancies: List[Tuple[int, int]] = []
    row_gaps: List[Tuple[int, int]] = []
    for r in range(row_min, row_max):
        window = state[r, col_min:col_max]
        missing = int((col_max - col_min) - np.sum(window))
        if missing > 0:
            row_gaps.append((missing, r))
    row_gaps.sort(key=lambda item: item[0], reverse=True)
    for _, r in row_gaps:
        for c in range(col_min, col_max):
            if state[r, c] == 0:
                vacancies.append((r, c))
    return vacancies


def _choose_sequence(
    state: np.ndarray,
    donor_row: int,
    donor_col: int,
    target_row: int,
    target_col: int,
) -> List[Tuple[str, Move]] | None:
    if donor_col == target_col and _column_path_clear(
        state, donor_col, donor_row, target_row
    ):
        return [("vertical", Move(donor_row, donor_col, target_row, donor_col))]
    if donor_row == target_row and _row_path_clear(
        state, donor_row, donor_col, target_col
    ):
        return [("horizontal", Move(donor_row, donor_col, target_row, target_col))]

    # For 2-step moves, we must ensure the intermediate spot is empty.
    # Option 1: Horizontal then Vertical. Intermediate: (donor_row, target_col)
    if (
        state[donor_row, target_col] == 0
        and _row_path_clear(state, donor_row, donor_col, target_col)
        and _column_path_clear(state, target_col, donor_row, target_row)
    ):
        return [
            ("horizontal", Move(donor_row, donor_col, donor_row, target_col)),
            ("vertical", Move(donor_row, target_col, target_row, target_col)),
        ]

    # Option 2: Vertical then Horizontal. Intermediate: (target_row, donor_col)
    if (
        state[target_row, donor_col] == 0
        and _column_path_clear(state, donor_col, donor_row, target_row)
        and _row_path_clear(state, target_row, donor_col, target_col)
    ):
        return [
            ("vertical", Move(donor_row, donor_col, target_row, donor_col)),
            ("horizontal", Move(target_row, donor_col, target_row, target_col)),
        ]
    return None


def _plan_gap_fill(
    state: np.ndarray,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    dop: int | None,
) -> Tuple[List[List[Move]], bool]:
    """Stage 3: prioritized gap filling with donor-aware ordering."""
    if row_min >= row_max or col_min >= col_max:
        return [], True
    donors = _collect_donors_by_row(state, row_min, row_max, col_min, col_max)
    result_batches: List[List[Move]] = []
    current_axis: str | None = None
    current_batch: List[Move] = []

    def flush(axis: str | None = None) -> None:
        nonlocal result_batches, current_batch, current_axis
        if not current_batch:
            return
        result_batches.extend(_chunk_batch(current_batch, dop))
        current_batch = []
        current_axis = axis

    while True:
        vacancies = _ordered_vacancies(state, row_min, row_max, col_min, col_max)
        if not vacancies:
            break
        if not donors:
            flush()
            return result_batches, False
        progress = False
        for vr, vc in vacancies:
            donor_rows = sorted(
                donors.keys(), key=lambda r: len(donors[r]), reverse=True
            )
            chosen: List[Tuple[str, Move]] | None = None
            selected_row = -1
            selected_col = -1
            for dr in donor_rows:
                columns = sorted(
                    donors[dr], key=lambda col: (abs(col - vc), -len(donors[dr]))
                )
                for dc in columns:
                    sequence = _choose_sequence(state, dr, dc, vr, vc)
                    if sequence is None:
                        continue
                    chosen = sequence
                    selected_row = dr
                    selected_col = dc
                    break
                if chosen is not None:
                    break
            if chosen is None:
                continue
            progress = True
            donors[selected_row].remove(selected_col)
            if not donors[selected_row]:
                donors.pop(selected_row, None)
            for axis, move in chosen:
                if current_axis is None:
                    current_axis = axis
                if axis != current_axis:
                    flush(axis)
                if current_batch:
                    conflict = False
                    # Check for hardware conflict: same source cannot map to different targets in one batch
                    if axis == "vertical":
                        for m in current_batch:
                            if m.from_row == move.from_row and m.to_row != move.to_row:
                                conflict = True
                                break
                    elif axis == "horizontal":
                        for m in current_batch:
                            if m.from_col == move.from_col and m.to_col != move.to_col:
                                conflict = True
                                break

                    if conflict:
                        flush(axis)

                current_batch.append(move)
                _apply_batches_to_state(state, [[move]])
                if dop is not None and dop > 0 and len(current_batch) >= dop:
                    flush(current_axis)
        if not progress:
            break
    flush()
    finished = not _ordered_vacancies(state, row_min, row_max, col_min, col_max)
    return result_batches, finished


def pcfa_algorithm(
    initial: np.ndarray,
    target: np.ndarray,
    dop: int | None = None,
) -> Tuple[np.ndarray, List[List[Move]], bool]:
    """Top-level PCFA orchestration aligned with the published protocol."""
    state = np.array(initial.copy(), dtype=int)
    target_mask = (np.array(target, dtype=int) > 0).astype(int)
    if target_mask.shape != state.shape:
        raise ValueError("Target mask must match initial state dimensions for PCFA.")
    if not np.any(target_mask):
        return state, [], True
    available_atoms = int(np.sum(state))
    required_atoms = int(np.sum(target_mask))
    if available_atoms < required_atoms:
        raise ValueError(
            "Not enough atoms in initial configuration to reach target mask."
        )
    row_min, row_max, col_min, col_max = _target_bounds_from_mask(target_mask)
    move_plan: List[List[Move]] = []
    stage2 = _plan_row_compression(
        state, row_min, row_max, col_min, col_max - col_min, dop
    )
    move_plan.extend(stage2)
    stage3, success_fill = _plan_gap_fill(
        state, row_min, row_max, col_min, col_max, dop
    )
    move_plan.extend(stage3)
    success_flag = success_fill and np.array_equal(state * target_mask, target_mask)
    return state, move_plan, success_flag
