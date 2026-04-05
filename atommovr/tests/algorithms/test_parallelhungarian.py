import numpy as np
import pytest

from atommovr.algorithms.source import Hungarian_works as hw
from atommovr.utils.move_utils import (
    Move,
    find_destructive_support_mask_from_moves,
    move_atoms_noiseless,
    get_AOD_cmds_from_move_list,
)


def _move_atoms_2d_compat(
    init_matrix: np.ndarray,
    moves: list[Move],
):
    """
    Minimal deterministic 2D single-species move application for Hungarian
    scheduling regression tests.

    Parameters
    ----------
    init_matrix : np.ndarray
        Binary occupancy matrix of shape (rows, cols).
    moves : list[Move]
        Sequential move list to apply.

    Returns
    -------
    tuple[np.ndarray, list]
        Updated matrix and an empty failure list, matching the call shape used
        by Hungarian_works.
    """
    matrix_out = init_matrix.copy()

    try:
        moves[0]
    except TypeError:
        moves = [moves]

    for move in moves:
        if matrix_out[move.from_row, move.from_col] != 1:
            continue

        matrix_out[move.from_row, move.from_col] = 0
        matrix_out[move.to_row, move.to_col] = 1

    return matrix_out, []


def _serialize_parallel_move_set(
    parallel_move_set: list[list[Move]],
) -> list[list[tuple[int, int, int, int]]]:
    """
    Convert grouped Move objects to plain tuples for stable equality checks.
    """
    serialized: list[list[tuple[int, int, int, int]]] = []
    for move_group in parallel_move_set:
        serialized.append(
            [
                (move.from_row, move.from_col, move.to_row, move.to_col)
                for move in move_group
            ]
        )
    return serialized


def _copy_path_structure(
    paths: list[list[list[Move]]],
) -> list[list[list[Move]]]:
    """
    Make a structural copy of the nested path container without cloning Move
    objects, which are treated as immutable test fixtures here.
    """
    return [[list(move_box) for move_box in path] for path in paths]


def _serialize_parallel_groups(
    groups: list[list[Move]],
) -> list[list[tuple[int, int, int, int]]]:
    """
    Convert grouped Move objects into plain tuples for equality checks.
    """
    return [
        [(move.from_row, move.from_col, move.to_row, move.to_col) for move in group]
        for group in groups
    ]


def _apply_groups_noiseless(
    matrix: np.ndarray,
    groups: list[list[Move]],
) -> np.ndarray:
    """
    Apply a grouped move schedule with the deterministic planning helper.
    """
    out = matrix.copy()
    for group in groups:
        out = move_atoms_noiseless(out, group)
    return out


def _apply_moves_sequential_noiseless(
    matrix: np.ndarray,
    moves: list[Move],
) -> np.ndarray:
    """
    Apply a move list one move at a time with the deterministic planning helper.
    """
    out = matrix.copy()
    for move in moves:
        out = move_atoms_noiseless(out, [move])
    return out


class TestREGROUP_PARALLEL_MOVES:
    def test_groups_noninterfering_moves_into_valid_batches(self) -> None:
        """
        Noninterfering moves should be grouped into physically valid parallel
        batches, and grouped execution should match sequential execution.
        """
        matrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        moves = [
            Move(0, 0, 0, 1),
            Move(0, 2, 0, 3),
        ]

        groups = hw.regroup_parallel_moves_fast(matrix.copy(), list(moves))

        assert len(groups) == 1

        for group in groups:
            support_mask, ok = find_destructive_support_mask_from_moves(matrix, group)
            assert ok
            assert not support_mask.any()

        grouped_final = _apply_groups_noiseless(matrix.copy(), groups)
        sequential_final = _apply_moves_sequential_noiseless(matrix.copy(), moves)

        assert np.array_equal(grouped_final, sequential_final)

    def test_grouped_execution_matches_sequential_execution_on_mixed_case(self) -> None:
        """
        Grouped execution should preserve the noiseless final state even when
        the new batching differs from the legacy grouping.
        """
        matrix = np.array(
            [
                [1, 0, 1],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        )
        moves = [
            Move(0, 0, 1, 0),
            Move(0, 2, 1, 2),
            Move(1, 1, 2, 1),
        ]

        groups = hw.regroup_parallel_moves_fast(matrix.copy(), list(moves))

        for group in groups:
            support_mask, ok = find_destructive_support_mask_from_moves(matrix, group)
            assert ok
            assert not support_mask.any()

        grouped_final = _apply_groups_noiseless(matrix.copy(), groups)
        sequential_final = _apply_moves_sequential_noiseless(matrix.copy(), moves)

        assert np.array_equal(grouped_final, sequential_final)

    def test_grouped_execution_matches_sequential_execution_on_random_small_cases(
        self,
    ) -> None:
        """
        On small generated move sets, every returned batch should be physically
        admissible and grouped execution should match sequential execution.
        """
        rng = np.random.default_rng(0)

        for side in [4, 5]:
            for _ in range(20):
                matrix = (rng.random((side, side)) < 0.25).astype(np.uint8, copy=False)
                target = (rng.random((side, side)) < 0.25).astype(np.uint8, copy=False)

                prepared_assignments = hw.generate_assignments(
                    matrix.copy(), target.copy(), []
                )
                candidate_moves: list[Move] = []

                for start, end in prepared_assignments[
                    : min(4, len(prepared_assignments))
                ]:
                    path = hw.generate_path(matrix.copy(), start, end)
                    if path:
                        for boxed_move in path:
                            candidate_moves.append(boxed_move[0])

                groups = hw.regroup_parallel_moves_fast(
                    matrix.copy(), list(candidate_moves)
                )

                for group in groups:
                    support_mask, ok = find_destructive_support_mask_from_moves(
                        matrix, group
                    )
                    assert ok
                    assert not support_mask.any()

                grouped_final = _apply_groups_noiseless(matrix.copy(), groups)
                sequential_final = _apply_moves_sequential_noiseless(
                    matrix.copy(), candidate_moves
                )

                assert np.array_equal(grouped_final, sequential_final)

    def test_regroup_adds_ghost_moves_for_full_tone_product(self) -> None:
        """
        Parallel regrouping should emit ghost moves so every active AOD tone
        combination is represented in the move list within each group.
        """
        matrix = np.zeros((3, 4), dtype=np.uint8)
        matrix[0, 0] = 1
        matrix[0, 2] = 1
        matrix[1, 1] = 1

        moves = [
            Move(0, 0, 0, 1),  # horizontal +1
            Move(1, 1, 2, 1),  # vertical +1
            Move(0, 2, 0, 3),  # horizontal +1
        ]

        groups = hw.regroup_parallel_moves_fast(matrix.copy(), list(moves))

        # The greedy algorithm may produce multiple groups depending on move compatibility
        assert len(groups) > 0, "Should produce at least one group"

        # Verify each group has canonicalization applied - no group should be empty
        for group in groups:
            assert len(group) > 0, "Each group should have moves"

        # Collect all moves across all groups
        all_regrouped = []
        for group in groups:
            all_regrouped.extend(group)

        # Get AOD commands from original moves to determine expected tone product
        horiz_cmds, vert_cmds, ok = get_AOD_cmds_from_move_list(matrix, moves)
        assert ok

        # Within each group, verify canonicalization added ghost moves appropriately
        for group in groups:
            group_horiz, group_vert, _ = get_AOD_cmds_from_move_list(matrix, group)

            # If group has both horizontal and vertical moves, check for tone product fills
            if group_horiz.any() and group_vert.any():
                h_active = np.count_nonzero(group_horiz)
                v_active = np.count_nonzero(group_vert)
                # Each group with both horizontal and vertical should have h_active * v_active moves
                expected_in_group = h_active * v_active
                assert len(group) == expected_in_group, (
                    f"Group with {h_active} horiz and {v_active} vert tones should have "
                    f"{expected_in_group} moves (full tone product), got {len(group)}"
                )

        # Verify all original moves are preserved across all groups
        original_serialized = {
            (move.from_row, move.from_col, move.to_row, move.to_col) for move in moves
        }
        all_regrouped_serialized = {
            (move.from_row, move.from_col, move.to_row, move.to_col)
            for move in all_regrouped
        }
        assert all_regrouped_serialized.issuperset(
            original_serialized
        ), "All original moves should be preserved in regrouped batches"

    # def test_matches_original_on_noninterfering_moves(self) -> None:
    #     """
    #     Behavioral regression test: fast regrouping must match the original on a
    #     simple set of compatible moves.
    #     """
    #     matrix = np.array(
    #         [
    #             [1, 0, 1, 0],
    #             [0, 0, 0, 0],
    #             [0, 0, 0, 0],
    #         ],
    #         dtype=np.uint8,
    #     )
    #     moves = [
    #         Move(0, 0, 0, 1),
    #         Move(0, 2, 0, 3),
    #     ]

    #     ref_groups = hw.regroup_parallel_moves(matrix.copy(), list(moves))
    #     new_groups = hw.regroup_parallel_moves_fast(matrix.copy(), list(moves))

    #     assert _serialize_parallel_groups(new_groups) == _serialize_parallel_groups(ref_groups)

    # def test_matches_original_on_mixed_compatibility_moves(self) -> None:
    #     """
    #     Behavioral regression test: fast regrouping must preserve greedy batch
    #     formation when only some candidates can join a batch.
    #     """
    #     matrix = np.array(
    #         [
    #             [1, 0, 1],
    #             [0, 1, 0],
    #             [0, 0, 0],
    #         ],
    #         dtype=np.uint8,
    #     )
    #     moves = [
    #         Move(0, 0, 1, 0),
    #         Move(0, 2, 1, 2),
    #         Move(1, 1, 2, 1),
    #     ]

    #     ref_groups = hw.regroup_parallel_moves(matrix.copy(), list(moves))
    #     new_groups = hw.regroup_parallel_moves_fast(matrix.copy(), list(moves))

    #     assert _serialize_parallel_groups(new_groups) == _serialize_parallel_groups(ref_groups)

    # def test_matches_original_on_random_small_cases(self) -> None:
    #     """
    #     Behavioral regression test: fast regrouping must match the original on
    #     small move lists generated from existing path construction.
    #     """
    #     rng = np.random.default_rng(0)

    #     for side in [4, 5]:
    #         for _ in range(20):
    #             matrix = (rng.random((side, side)) < 0.25).astype(np.uint8, copy=False)
    #             target = (rng.random((side, side)) < 0.25).astype(np.uint8, copy=False)

    #             prepared_assignments = hw.generate_assignments(matrix.copy(), target.copy(), [])
    #             candidate_moves: list[Move] = []

    #             for start, end in prepared_assignments[: min(4, len(prepared_assignments))]:
    #                 path = hw.generate_path(matrix.copy(), start, end)
    #                 if path:
    #                     for boxed_move in path:
    #                         candidate_moves.append(boxed_move[0])

    #             ref_groups = hw.regroup_parallel_moves(matrix.copy(), list(candidate_moves))
    #             new_groups = hw.regroup_parallel_moves_fast(matrix.copy(), list(candidate_moves))

    #             assert _serialize_parallel_groups(new_groups) == _serialize_parallel_groups(ref_groups)


class TestTRANSFORM_PATHS_INTO_MOVES:
    @pytest.fixture(autouse=True)
    def _patch_move_atoms(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Patch Hungarian_works.move_atoms with a lightweight 2D-compatible
        deterministic move applier so the regression tests isolate scheduling
        behavior from unrelated move_utils API changes.
        """
        monkeypatch.setattr(hw, "move_atoms", _move_atoms_2d_compat)

    def test_matches_final_state_on_handcrafted_nonintersecting_paths(self) -> None:
        """
        The fast transform should preserve the final state; exact batch
        boundaries are allowed to differ from the legacy implementation.
        """
        matrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

        paths = [
            [
                [Move(0, 0, 0, 1)],
                [Move(0, 1, 1, 1)],
            ],
            [
                [Move(0, 2, 0, 3)],
                [Move(0, 3, 1, 3)],
            ],
        ]

        ref_matrix, ref_moves = hw.transform_paths_into_moves(
            matrix.copy(),
            _copy_path_structure(paths),
        )
        new_matrix, new_moves = hw.transform_paths_into_moves_fast(
            matrix.copy(),
            _copy_path_structure(paths),
        )

        assert np.array_equal(new_matrix, ref_matrix)

        for group in new_moves:
            support_mask, ok = find_destructive_support_mask_from_moves(matrix, group)
            assert ok
            assert not support_mask.any()

    def test_matches_final_state_on_generated_paths_small_random_cases(self) -> None:
        """
        The fast transform should preserve the final state on generated path
        sets, even if the returned batching is more compact than the legacy
        batching.
        """
        rng = np.random.default_rng(0)

        for side in [4, 5]:
            for _ in range(20):
                matrix = (rng.random((side, side)) < 0.25).astype(np.uint8, copy=False)
                target = (rng.random((side, side)) < 0.25).astype(np.uint8, copy=False)

                prepared_assignments = hw.generate_assignments(
                    matrix.copy(), target.copy(), []
                )
                paths: list[list[list[Move]]] = []

                for start, end in prepared_assignments[
                    : min(4, len(prepared_assignments))
                ]:
                    path = hw.generate_path(matrix.copy(), start, end)
                    if path != []:
                        paths.append(path)

                ref_matrix, ref_moves = hw.transform_paths_into_moves(
                    matrix.copy(),
                    _copy_path_structure(paths),
                )
                new_matrix, new_moves = hw.transform_paths_into_moves_fast(
                    matrix.copy(),
                    _copy_path_structure(paths),
                )

                assert np.array_equal(new_matrix, ref_matrix)

                for group in new_moves:
                    support_mask, ok = find_destructive_support_mask_from_moves(
                        matrix, group
                    )
                    assert ok
                    assert not support_mask.any()

                assert len(new_moves) <= len(ref_moves)

    def test_matches_original_on_handcrafted_intersecting_paths(self) -> None:
        """
        Behavioral regression test: refactored scheduling must preserve the
        original intersection-delay behavior.
        """
        matrix = np.array(
            [
                [1, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        )

        paths = [
            [
                [Move(0, 0, 0, 1)],
                [Move(0, 1, 1, 1)],
            ],
            [
                [Move(0, 2, 0, 1)],
                [Move(0, 1, 1, 1)],
            ],
        ]

        ref_matrix, ref_moves = hw.transform_paths_into_moves(
            matrix.copy(),
            _copy_path_structure(paths),
        )
        new_matrix, new_moves = hw.transform_paths_into_moves_fast(
            matrix.copy(),
            _copy_path_structure(paths),
        )

        assert np.array_equal(new_matrix, ref_matrix)
        assert _serialize_parallel_move_set(new_moves) == _serialize_parallel_move_set(
            ref_moves
        )

    def test_matches_original_on_empty_input(self) -> None:
        """
        Edge-case regression test: empty path collections should remain a
        no-op.
        """
        matrix = np.zeros((4, 4), dtype=np.uint8)
        paths: list[list[list[Move]]] = []

        ref_matrix, ref_moves = hw.transform_paths_into_moves(
            matrix.copy(),
            _copy_path_structure(paths),
        )
        new_matrix, new_moves = hw.transform_paths_into_moves_fast(
            matrix.copy(),
            _copy_path_structure(paths),
        )

        assert np.array_equal(new_matrix, ref_matrix)
        assert _serialize_parallel_move_set(new_moves) == _serialize_parallel_move_set(
            ref_moves
        )
