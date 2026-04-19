"""Tetris atom-rearrangement algorithm.

This module implements the row-by-row "Tetrimino construction" and column-by-column
"Tetrimino elimination" phases described in Wang et al. [1]_.
The implementation mirrors the hardware-friendly workflow:

1. Row processing assigns atoms from each reservoir row to the columns that still
    require population, prioritizing the smallest outstanding target-row indices
    globally. The assignment keeps the relative ordering of atoms intact so that each
    row activation (one batch) can move all atoms in that row independently without
    tweezer crossings.
2. Column compression reshuffles the atoms inside each target column onto their
    desired row indices, emitting exactly one batch per column activation because the
    hardware cannot drive multiple columns simultaneously.

Both stages emit axis-aligned lists of ``Move`` objects that can be executed by
``AtomArray.evaluate_moves``.
"""

from __future__ import annotations

# Copyright (c) 2026 Sascha Benz
# Ludwig-Maximilians-Universitaet Muenchen (Ryd-Yb tweezer array group)
# SPDX-License-Identifier: MIT
#
# Implementation author: Sascha Benz
# This file implements the Tetris algorithm described in:
# Wang, S., Zhang, W., Zhang, T., Mei, S., Wang, Y., Hu, J., Chen, W. (2023).
# Accelerating the Assembly of Defect-Free Atomic Arrays with Maximum Parallelisms.
# Phys. Rev. Applied 19, 054032. DOI: 10.1103/PhysRevApplied.19.054032
#

from collections import deque
from typing import List, Sequence, Tuple

import numpy as np

from atommovr.utils.Move import Move

MoveBatches = List[List[Move]]


def _row_rearrangement(
    state: np.ndarray,
    column_requirements: List[deque[int]],
) -> Tuple[MoveBatches, bool]:
    """Perform Tetrimino construction (row-by-row horizontal moves)."""
    batches: MoveBatches = []
    n_rows = state.shape[0]
    remaining = sum(len(req) for req in column_requirements)

    for row in range(n_rows):
        if remaining == 0:
            break
        src_cols = list(np.where(state[row, :] == 1)[0])
        if not src_cols:
            continue

        priorities = sorted(
            (req[0], col) for col, req in enumerate(column_requirements) if req
        )
        if not priorities:
            break

        selected_cols = [col for _, col in priorities[: len(src_cols)]]
        if not selected_cols:
            continue

        # Determine targets for this row
        desired_cols = sorted(selected_cols)
        current_cols = sorted(src_cols)

        # Update requirements immediately (reservation)
        for tgt_col in selected_cols:
            if column_requirements[tgt_col]:
                column_requirements[tgt_col].popleft()
                remaining -= 1

        # Use monotonic matching to assign sources to targets
        pairs = _match_monotonic(current_cols, desired_cols)

        row_moves: List[Move] = []
        used_sources = set()

        for src_col, tgt_col in pairs:
            used_sources.add(src_col)
            if src_col == tgt_col:
                continue
            row_moves.append(Move(row, src_col, row, tgt_col))

        # Update state
        # Clear all sources
        for s in current_cols:
            state[row, s] = 0
        # Set targets
        for _, t in pairs:
            state[row, t] = 1
        # Restore unused sources
        for s in current_cols:
            if s not in used_sources:
                state[row, s] = 1

        if row_moves:
            batches.append(row_moves)

    success = remaining == 0
    return batches, success


def _match_monotonic(sources: List[int], targets: List[int]) -> List[Tuple[int, int]]:
    """Find a monotonic matching between a subset of sources and targets that minimizes displacement."""
    n = len(sources)
    m = len(targets)

    # dp[i][j] = min cost to match first j targets using a subset of first i sources
    # Initialize with infinity
    dp = np.full((n + 1, m + 1), float("inf"))

    # Base case: 0 targets matched with 0 sources cost 0
    for i in range(n + 1):
        dp[i][0] = 0.0

    # Fill DP
    for j in range(1, m + 1):
        for i in range(j, n + 1):
            cost = abs(sources[i - 1] - targets[j - 1])
            match_cost = dp[i - 1][j - 1] + cost
            skip_cost = dp[i - 1][j]
            dp[i][j] = min(match_cost, skip_cost)

    # Backtrack
    matches = []
    i, j = n, m
    while j > 0:
        cost = abs(sources[i - 1] - targets[j - 1])
        # Check if we came from match or skip
        # Use a small epsilon for float comparison if needed, but here costs are integers (or close enough)
        if dp[i][j] == dp[i - 1][j - 1] + cost:
            matches.append((sources[i - 1], targets[j - 1]))
            i -= 1
            j -= 1
        else:
            i -= 1

    return matches[::-1]


def _compress_columns(
    state: np.ndarray,
    column_targets: Sequence[Sequence[int]],
) -> Tuple[MoveBatches, bool]:
    """Perform Tetrimino elimination (column-wise vertical compression).

    Each column's moves form a single batch because a column activation corresponds
    to exactly one hardware command sequence on the vertical AOD axis.
    """
    batches: MoveBatches = []
    for col, target_rows in enumerate(column_targets):
        if not target_rows:
            continue
        src_rows = sorted(np.where(state[:, col] == 1)[0])
        sorted_targets = sorted(target_rows)

        if len(src_rows) < len(sorted_targets):
            print(
                f"Not enough atoms to fill column {col}: have {len(src_rows)}, need {len(sorted_targets)}"
            )
            return batches, False

        # Find optimal monotonic matching
        pairs = _match_monotonic(src_rows, sorted_targets)

        col_moves: List[Move] = []
        used_sources = set()

        for src_row, tgt_row in pairs:
            used_sources.add(src_row)
            if src_row == tgt_row:
                continue
            col_moves.append(Move(src_row, col, tgt_row, col))

        # Update state
        # Clear all sources
        for s in src_rows:
            state[s, col] = 0
        # Set targets
        for _, t in pairs:
            state[t, col] = 1
        # Restore unused sources
        for s in src_rows:
            if s not in used_sources:
                state[s, col] = 1

        if col_moves:
            batches.append(col_moves)
    return batches, True


def tetris_algorithm(
    initial: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, MoveBatches, bool]:
    """Plan Tetris rearrangement moves."""
    state = np.array(initial, dtype=int)
    target_mask = (np.array(target, dtype=int) > 0).astype(int)
    required_atoms = int(np.sum(target_mask))
    if required_atoms == 0:
        return state, [], True
    if int(np.sum(state)) < required_atoms:
        raise ValueError("Not enough atoms in initial configuration to reach target.")

    target_sites = target_mask.astype(bool)

    column_targets = [
        list(np.where(target_mask[:, col] == 1)[0])
        for col in range(target_mask.shape[1])
    ]
    column_requirements = [deque(rows) for rows in column_targets]
    row_batches, constructed = _row_rearrangement(state, column_requirements)

    if np.all(state[target_sites] == 1):
        # Target already realized; column phase unnecessary.
        return state, row_batches, True

    column_batches, compressed = _compress_columns(state, column_targets)
    move_plan: MoveBatches = row_batches + column_batches
    success = bool(constructed and compressed and np.all(state[target_sites] == 1))
    return state, move_plan, success
