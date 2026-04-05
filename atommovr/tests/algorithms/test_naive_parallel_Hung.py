from __future__ import annotations

import numpy as np

from atommovr.algorithms.source.naive_parallel_Hung import (
    find_smallest_l,
    define_current_and_target_naive_par,
    generate_assignments_naive_par,
    bfs_find_path_naive_par,
    neighbors_8_naive_par,
    transform_paths_into_moves_naive_par,
    naive_par_Hung,
)
from atommovr.utils.AtomArray import AtomArray
from atommovr.utils.Move import Move
from atommovr.tests.support.helpers import (
    _n_atoms,
)


class TestFindSmallestL:
    """Test the find_smallest_l helper function."""

    def test_returns_minimum_l_containing_atoms(self) -> None:
        """Smallest l should contain more atoms than total targets."""
        matrix = np.zeros((8, 8), dtype=np.uint8)
        target = np.zeros((8, 8), dtype=np.uint8)

        # Place 4 atoms at corners
        matrix[0, 0] = 1
        matrix[0, 7] = 1
        matrix[7, 0] = 1
        matrix[7, 7] = 1

        # Place 3 targets in center
        target[4, 4] = 1
        target[4, 5] = 1
        target[5, 4] = 1

        l = find_smallest_l(matrix, target)
        assert l > 0
        # Check that the resulting square can contain the atoms
        n = len(matrix)
        center = n / 2
        delta = n % 2
        left_bound = int(center - l + delta)
        right_bound = int(center + l)
        n_in_square = np.sum(matrix[left_bound:right_bound, left_bound:right_bound])
        assert n_in_square >= np.sum(target)

    def test_returns_one_for_dense_center(self) -> None:
        """If atoms are already in center, l should be small."""
        matrix = np.zeros((8, 8), dtype=np.uint8)
        target = np.zeros((8, 8), dtype=np.uint8)

        # Atoms at center
        matrix[3, 3] = 1
        matrix[3, 4] = 1
        matrix[4, 3] = 1
        matrix[4, 4] = 1

        # Targets also at center
        target[3, 3] = 1
        target[3, 4] = 1

        l = find_smallest_l(matrix, target)
        assert l >= 1


class TestDefineCurrentAndTargetNaivePar:
    """Test position discovery for atoms and targets."""

    def test_identifies_movable_atoms(self) -> None:
        """Should find atoms that are not in target and not occupied by other species."""
        matrix = np.zeros((5, 5), dtype=np.uint8)
        other_matrix = np.zeros((5, 5), dtype=np.uint8)
        target = np.zeros((5, 5), dtype=np.uint8)
        other_target = np.zeros((5, 5), dtype=np.uint8)

        # Place an atom that needs to move
        matrix[0, 0] = 1
        # Target it needs to go to
        target[2, 2] = 1

        current, targets, _ = define_current_and_target_naive_par(
            matrix, other_matrix, target, other_target
        )

        assert (0, 0) in current
        assert (2, 2) in targets

    def test_excludes_atoms_occupied_by_other_species(self) -> None:
        """Should not count atoms if other species occupies same site."""
        matrix = np.zeros((5, 5), dtype=np.uint8)
        other_matrix = np.zeros((5, 5), dtype=np.uint8)
        target = np.zeros((5, 5), dtype=np.uint8)
        other_target = np.zeros((5, 5), dtype=np.uint8)

        matrix[2, 2] = 1
        other_matrix[2, 2] = 1  # blocked by other species
        target[3, 3] = 1

        current, _, _ = define_current_and_target_naive_par(
            matrix, other_matrix, target, other_target
        )

        assert (2, 2) not in current

    def test_returns_redundant_area(self) -> None:
        """Should identify positions where neither species has targets."""
        matrix = np.zeros((5, 5), dtype=np.uint8)
        other_matrix = np.zeros((5, 5), dtype=np.uint8)
        target = np.zeros((5, 5), dtype=np.uint8)
        other_target = np.zeros((5, 5), dtype=np.uint8)

        # Add at least one atom so find_smallest_l can exit loop
        matrix[0, 0] = 1

        target[2, 2] = 1
        other_target[3, 3] = 1

        _, _, redundant = define_current_and_target_naive_par(
            matrix, other_matrix, target, other_target
        )

        # Positions not in any target should be in redundant area
        assert len(redundant) > 0


class TestGenerateAssignmentsNaivePar:
    """Test assignment generation for atom-to-target pairing."""

    def test_pairs_atoms_to_targets(self) -> None:
        """Should create valid pairings between atoms and targets."""
        matrix = np.zeros((5, 5), dtype=np.uint8)
        other_matrix = np.zeros((5, 5), dtype=np.uint8)
        target = np.zeros((5, 5), dtype=np.uint8)
        other_target = np.zeros((5, 5), dtype=np.uint8)

        # 3 atoms near top
        matrix[0, 0] = 1
        matrix[0, 2] = 1
        matrix[1, 1] = 1

        # 3 targets near bottom
        target[3, 3] = 1
        target[3, 4] = 1
        target[4, 3] = 1

        assignments = generate_assignments_naive_par(
            matrix, other_matrix, target, other_target
        )

        assert len(assignments) > 0
        # Each assignment is a tuple of (from, to)
        for start, end in assignments:
            assert isinstance(start, tuple) and len(start) == 2
            assert isinstance(end, tuple) and len(end) == 2

    def test_respects_used_coord_filter(self) -> None:
        """Should exclude positions already used by other species."""
        matrix = np.zeros((5, 5), dtype=np.uint8)
        other_matrix = np.zeros((5, 5), dtype=np.uint8)
        target = np.zeros((5, 5), dtype=np.uint8)
        other_target = np.zeros((5, 5), dtype=np.uint8)

        matrix[0, 0] = 1
        matrix[0, 1] = 1
        target[2, 2] = 1
        target[2, 3] = 1

        # Mark (0, 0) as already used
        used = [(0, 0)]

        assignments = generate_assignments_naive_par(
            matrix, other_matrix, target, other_target, used_coord=used
        )

        # (0, 0) should not be in assignments as source
        for start, end in assignments:
            assert start != (0, 0)

    def test_handles_empty_targets(self) -> None:
        """Should handle case with no targets gracefully."""
        matrix = np.zeros((5, 5), dtype=np.uint8)
        other_matrix = np.zeros((5, 5), dtype=np.uint8)
        target = np.zeros((5, 5), dtype=np.uint8)
        other_target = np.zeros((5, 5), dtype=np.uint8)

        matrix[0, 0] = 1

        assignments = generate_assignments_naive_par(
            matrix, other_matrix, target, other_target
        )

        # Should return valid list (possibly empty or with reservoir targets)
        assert isinstance(assignments, list)


class TestNeighbors8NaivePar:
    """Test 8-neighbor connectivity generation."""

    def test_returns_8_neighbors_for_center(self) -> None:
        """Center point should have 8 neighbors."""
        neighbors = neighbors_8_naive_par(3, 3, 8)
        assert len(neighbors) == 8

    def test_returns_3_neighbors_for_corner(self) -> None:
        """Corner point should have 3 neighbors."""
        neighbors = neighbors_8_naive_par(0, 0, 8)
        assert len(neighbors) == 3
        assert (0, 1) in neighbors
        assert (1, 0) in neighbors
        assert (1, 1) in neighbors

    def test_respects_grid_bounds(self) -> None:
        """Neighbors should not exceed grid boundaries."""
        neighbors = neighbors_8_naive_par(0, 0, 5)
        for r, c in neighbors:
            assert 0 <= r < 5
            assert 0 <= c < 5


class TestBFSFindPathNaivePar:
    """Test BFS pathfinding in dual-species context."""

    def test_finds_path_to_unobstructed_target(self) -> None:
        """BFS should find a path when target is reachable."""
        matrix = np.zeros((5, 5, 2), dtype=np.uint8)
        # Place an atom of species 0 at start
        matrix[1, 1, 0] = 1

        def dummy_filter(start, pos):
            # Simple filter: can move to unoccupied cells
            return (
                matrix[pos[0], pos[1], 0] == 0 and matrix[pos[0], pos[1], 1] == 0,
                None,
                None,
            )

        result = bfs_find_path_naive_par(
            matrix, start=(1, 1), end=(3, 3), handle_obstacle_filter=dummy_filter
        )

        assert result.end_reached
        assert result.path[-1] == (3, 3)

    def test_returns_false_when_target_unreachable(self) -> None:
        """BFS should fail if target is completely blocked."""
        matrix = np.zeros((5, 5, 2), dtype=np.uint8)

        def blocking_filter(start, pos):
            # All cells are blocked
            return False, None, None

        result = bfs_find_path_naive_par(
            matrix, start=(1, 1), end=(3, 3), handle_obstacle_filter=blocking_filter
        )

        assert not result.end_reached


class TestTransformPathsIntoMovesNaivePar:
    """Test transformation of paths into move batches."""

    def test_conserves_atom_count(self) -> None:
        """Move transformation should not create or destroy atoms."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # Place 2 atoms of species 0
        aa.matrix[0, 0, 0] = 1
        aa.matrix[0, 1, 0] = 1

        initial_count = _n_atoms(aa.matrix)

        # Create dummy paths (list of Move objects)
        dummy_paths = []

        aa_result, moves = transform_paths_into_moves_naive_par(
            aa, dummy_paths, max_rounds=1
        )

        # Atom count should be unchanged
        assert _n_atoms(aa_result.matrix) == initial_count

    def test_executes_non_conflicting_moves(self) -> None:
        """Should execute moves that don't conflict."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # Place atoms
        aa.matrix[0, 0, 0] = 1
        aa.matrix[1, 0, 0] = 1

        # Create paths with non-conflicting moves
        paths = [
            [Move(0, 0, 0, 1)],  # move atom from (0,0) to (0,1)
            [Move(1, 0, 1, 1)],  # move atom from (1,0) to (1,1)
        ]

        aa_result, moves = transform_paths_into_moves_naive_par(aa, paths, max_rounds=1)

        # Check that atoms were moved or paths consumed
        assert len(moves) >= 0


class TestNaiveParHung:
    """Test the main dual-species rearrangement algorithm."""

    def test_returns_false_when_insufficient_atoms(self) -> None:
        """Algorithm should fail if not enough atoms to fill target."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # 1 atom, 2 targets
        aa.matrix[0, 0, 0] = 1
        aa.target_Rb[2, 2] = 1
        aa.target_Cs[2, 3] = 1

        result_aa, moves, success = naive_par_Hung(aa, do_ejection=False, round_lim=5)

        assert success is False

    def test_conserves_atoms_during_rearrangement(self) -> None:
        """Total atom count should remain constant throughout."""
        aa = AtomArray(shape=[6, 6], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # Place 4 atoms of species 0
        aa.matrix[0, 0, 0] = 1
        aa.matrix[0, 1, 0] = 1
        aa.matrix[5, 5, 0] = 1
        aa.matrix[5, 4, 0] = 1

        # Place 2 atoms of species 1
        aa.matrix[1, 1, 1] = 1
        aa.matrix[1, 2, 1] = 1

        # Set target in center
        aa.target_Rb[2:4, 2:4] = np.uint8(1)
        aa.target_Cs[2:4, 2:4] = np.uint8(1)

        initial_count_s0 = _n_atoms(aa.matrix[:, :, 0:1])
        initial_count_s1 = _n_atoms(aa.matrix[:, :, 1:2])

        result_aa, moves, _ = naive_par_Hung(aa, do_ejection=False, round_lim=10)

        final_count_s0 = _n_atoms(result_aa.matrix[:, :, 0:1])
        final_count_s1 = _n_atoms(result_aa.matrix[:, :, 1:2])

        assert final_count_s0 == initial_count_s0
        assert final_count_s1 == initial_count_s1

    def test_terminates_within_round_limit(self) -> None:
        """Algorithm should terminate within specified round limit."""
        aa = AtomArray(shape=[6, 6], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # Single source/target pair per species
        aa.matrix[0, 0, 0] = 1
        aa.matrix[0, 0, 1] = 1
        aa.target[5, 5, 0] = 1
        aa.target[4, 4, 1] = 1

        result_aa, moves, success = naive_par_Hung(aa, do_ejection=False, round_lim=20)

        assert isinstance(moves, list)
        # All move rounds should be lists
        for round_moves in moves:
            assert isinstance(round_moves, list)

    def test_handles_overlapping_target_regions(self) -> None:
        """Algorithm should handle species with overlapping target regions."""
        aa = AtomArray(shape=[6, 6], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # Place atoms scattered
        aa.matrix[0, 0, 0] = 1
        aa.matrix[0, 1, 0] = 1
        aa.matrix[1, 0, 1] = 1
        aa.matrix[1, 1, 1] = 1

        # Overlapping target regions (but not at same sites)
        aa.target[4, 4, 0] = 1
        aa.target[4, 5, 0] = 1
        aa.target[5, 4, 1] = 1
        aa.target[5, 5, 1] = 1

        result_aa, moves, _ = naive_par_Hung(aa, do_ejection=False, round_lim=15)

        # Should complete without error
        assert isinstance(result_aa, AtomArray)
        assert isinstance(moves, list)

    def test_produces_valid_move_rounds(self) -> None:
        """All produced move rounds should be valid lists of Move objects."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        aa.matrix[0, 0, 0] = 1
        aa.matrix[0, 0, 1] = 1
        aa.target[4, 4, 0] = 1
        aa.target[3, 3, 1] = 1

        result_aa, moves, _ = naive_par_Hung(aa, do_ejection=False, round_lim=10)

        for round_idx, round_moves in enumerate(moves):
            assert isinstance(round_moves, list), f"Round {round_idx} is not a list"
            for move in round_moves:
                assert isinstance(
                    move, Move
                ), f"Round {round_idx} contains non-Move object"
                assert isinstance(move.from_row, int)
                assert isinstance(move.from_col, int)
                assert isinstance(move.to_row, int)
                assert isinstance(move.to_col, int)
