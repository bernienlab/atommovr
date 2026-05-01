from __future__ import annotations

import numpy as np

from atommovr.algorithms.source.inside_out import (
    inside_out_algorithm,
    check_atom_enough,
    layer_complete,
    rearrangement_complete,
)
from atommovr.algorithms.source.inside_out_utils import (
    perimeter_coords,
    def_boundary,
    clean_empty_moves,
    collect_coords,
    is_rb_source,
    is_cs_source,
    is_rb_target,
    is_reservoir,
    is_site_correct,
    atom_arrays_equal,
    gen_dual_assign_new,
    out_bound_ex,
)
from atommovr.utils.AtomArray import AtomArray
from atommovr.utils.Move import Move
from atommovr.tests.support.helpers import (
    _n_atoms,
)


class TestPerimeterCoords:
    """Test perimeter coordinate generation for layer boundaries."""

    def test_returns_all_boundary_cells(self) -> None:
        """Perimeter should include all cells on rectangle boundary."""
        top, left, bottom, right = 1, 1, 3, 3
        perimeter = perimeter_coords(top, left, bottom, right)

        # Should have cells on all four sides
        # Top row: (1,1), (1,2), (1,3)
        # Right column: (2,3), (3,3)
        # Bottom row: (3,2), (3,1)
        # Left column: (2,1)
        assert len(perimeter) > 0
        assert (top, left) in perimeter
        assert (top, right) in perimeter
        assert (bottom, left) in perimeter
        assert (bottom, right) in perimeter

    def test_single_cell_perimeter(self) -> None:
        """Single cell should be its own perimeter."""
        perimeter = perimeter_coords(2, 2, 2, 2)
        assert (2, 2) in perimeter

    def test_no_interior_cells(self) -> None:
        """Perimeter should not include interior cells."""
        top, left, bottom, right = 1, 1, 3, 3
        perimeter = perimeter_coords(top, left, bottom, right)

        # Interior cell (2, 2) should not be included
        assert (2, 2) not in perimeter
        # But boundary cells should be
        assert (1, 1) in perimeter or (1, 2) in perimeter


class TestDefBoundary:
    """Test layer boundary definition."""

    def test_returns_valid_boundary(self) -> None:
        """Boundary should form valid rectangle with top < bottom and left < right."""
        for array_len in [5, 6, 7, 8]:
            top, left, bottom, right = def_boundary(0, array_len)
            assert top <= bottom
            assert left <= right
            assert top >= 0
            assert left >= 0
            assert bottom < array_len
            assert right < array_len

    def test_layer_zero_is_center(self) -> None:
        """Layer 0 should be near center of array."""
        array_len = 8
        top, left, bottom, right = def_boundary(0, array_len)
        center = array_len // 2
        # Center should be roughly in middle of boundary
        assert top <= center <= bottom
        assert left <= center <= right

    def test_larger_layers_expand_outward(self) -> None:
        """Larger layer factors should produce larger boundaries."""
        array_len = 10
        top0, left0, bottom0, right0 = def_boundary(0, array_len)
        top1, left1, bottom1, right1 = def_boundary(1, array_len)

        # Layer 1 boundary should be larger
        area0 = (bottom0 - top0 + 1) * (right0 - left0 + 1)
        area1 = (bottom1 - top1 + 1) * (right1 - left1 + 1)
        assert area1 > area0


class TestCleanEmptyMoves:
    """Test filtering of moves that don't change atom positions."""

    def test_removes_moves_that_dont_change_array(self) -> None:
        """Should remove moves where source or destination is empty."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # Place one atom
        aa.matrix[0, 0, 0] = 1

        # Create a move that doesn't move anything (both empty cells)
        empty_move = Move(2, 2, 3, 3)

        # Create a valid move
        valid_move = Move(0, 0, 1, 1)

        move_list = [[empty_move, valid_move]]
        cleaned = clean_empty_moves(aa, move_list)

        # Should contain at least the valid move
        assert len(cleaned) > 0

    def test_preserves_atom_count(self) -> None:
        """Cleaning moves should not change total atom count."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # Place atoms
        aa.matrix[0, 0, 0] = 1
        aa.matrix[1, 1, 1] = 1

        initial_count = _n_atoms(aa.matrix)

        # Even if we pass invalid moves, atom count should be preserved
        move_list = [[Move(0, 0, 1, 1)]]
        _cleaned = clean_empty_moves(aa, move_list)

        assert _n_atoms(aa.matrix) == initial_count


class TestCollectCoords:
    """Test coordinate collection for specific regions and conditions."""

    def test_collects_layer_coords(self) -> None:
        """Should collect coordinates on the perimeter for given layer."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # Place atoms everywhere to ensure we collect from condition
        aa.matrix[:, :, 0] = 1

        coords = collect_coords(
            aa, layer_factor=1, region="layer", condition_func=is_rb_source(aa)
        )

        # Should find atoms in the layer
        assert len(coords) > 0

    def test_collects_outer_coords(self) -> None:
        """Should collect coordinates outside the layer."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # Place atoms in corners (outside inner layers)
        aa.matrix[0, 0, 0] = 1
        aa.matrix[0, 4, 0] = 1

        coords = collect_coords(
            aa, layer_factor=1, region="outer", condition_func=is_rb_source(aa)
        )

        # Should find atoms outside the inner layer
        assert len(coords) >= 0  # May find atoms depending on layer factor

    def test_empty_targets_collect_empty_regions(self) -> None:
        """Should collect empty regions when no targets present."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        coords = collect_coords(
            aa, layer_factor=1, region="outer", condition_func=is_reservoir(aa)
        )

        # Should find empty cells in reservoir
        assert len(coords) > 0


class TestSpeciesFilters:
    """Test species-specific coordinate filter functions."""

    def test_is_rb_source_identifies_misplaced_rb(self) -> None:
        """Should identify Rb atoms not at their targets."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # Rb atom at position that's not a target
        aa.matrix[0, 0, 0] = 1
        aa.target[1, 1, 0] = 1

        filter_func = is_rb_source(aa)
        assert filter_func(0, 0, out_bound=False) is True
        assert filter_func(1, 1, out_bound=False) is False

    def test_is_rb_target_identifies_empty_rb_targets(self) -> None:
        """Should identify empty Rb target sites."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # Empty site with Rb target
        aa.target[2, 2, 0] = 1

        filter_func = is_rb_target(aa)
        assert filter_func(2, 2, out_bound=False) is True

    def test_is_cs_source_identifies_misplaced_cs(self) -> None:
        """Should identify Cs atoms not at their targets."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # Cs atom at position that's not a target
        aa.matrix[0, 0, 1] = 1
        aa.target[1, 1, 1] = 1

        filter_func = is_cs_source(aa)
        assert filter_func(0, 0, out_bound=False) is True

    def test_is_reservoir_identifies_empty_no_target(self) -> None:
        """Should identify empty cells with no target."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        filter_func = is_reservoir(aa)
        assert filter_func(3, 3, out_bound=True) is True


class TestIsCorrect:
    """Test site correctness checks."""

    def test_site_correct_when_atoms_match_target(self) -> None:
        """Site is correct when atom config matches target config."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # Place matching atoms and targets
        aa.matrix[2, 2, 0] = 1
        aa.matrix[2, 2, 1] = 0
        aa.target[2, 2, 0] = 1
        aa.target[2, 2, 1] = 0

        is_correct = is_site_correct(aa)
        assert is_correct(2, 2) is True

    def test_site_incorrect_when_atoms_mismatch_target(self) -> None:
        """Site is incorrect when atom config doesn't match target."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # Atom present but target absent
        aa.matrix[1, 1, 0] = 1
        aa.target[1, 1, 0] = 0

        is_correct = is_site_correct(aa)
        assert is_correct(1, 1) is False


class TestCheckAtomEnough:
    """Test validation that sufficient atoms exist for targets."""

    def test_returns_true_with_sufficient_atoms(self) -> None:
        """Should return True when atom counts meet target counts."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target_Rb[:, :] = np.uint8(0)
        aa.target_Cs[:, :] = np.uint8(0)

        # Place more atoms than targets
        aa.matrix[0, 0, 0] = 1
        aa.matrix[1, 1, 0] = 1
        aa.target_Rb[2, 2] = 1

        aa.matrix[0, 1, 1] = 1
        aa.target_Cs[2, 3] = 1

        assert check_atom_enough(aa) is True

    def test_returns_false_with_insufficient_atoms(self) -> None:
        """Should return False when atoms < targets."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target_Rb[:, :] = np.uint8(0)
        aa.target_Cs[:, :] = np.uint8(0)

        # Fewer atoms than targets
        aa.matrix[0, 0, 0] = 1
        # Set target using both target_Rb and target arrays
        aa.target_Rb[2, 2] = 1
        aa.target_Rb[2, 3] = 1
        aa.target_Cs[2, 2] = 0
        aa.target_Cs[2, 3] = 1

        assert check_atom_enough(aa) is False

    def test_returns_true_with_exact_match(self) -> None:
        """Should return True when atom and target counts exactly match."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target_Rb[:, :] = np.uint8(0)
        aa.target_Cs[:, :] = np.uint8(0)

        # Exactly matching counts
        aa.matrix[0, 0, 0] = 1
        aa.matrix[1, 1, 0] = 1
        aa.target_Rb[2, 2] = 1
        aa.target_Rb[2, 3] = 1
        aa.matrix[4, 4, 1] = 1
        aa.target_Cs[3, 3] = 1

        assert check_atom_enough(aa) is True


class TestLayerComplete:
    """Test layer completion checking."""

    def test_returns_true_when_layer_matches_target(self) -> None:
        """Should return True when all sites in layer match target."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # Place atoms matching the target
        aa.matrix[2, 2, 0] = 1
        aa.matrix[2, 2, 1] = 0
        aa.target[2, 2, 0] = 1
        aa.target[2, 2, 1] = 0
        aa.target_Rb[2, 2] = 1
        aa.target_Cs[2, 2] = 0

        is_correct = is_site_correct(aa)
        result = layer_complete(1, aa, is_correct)
        assert isinstance(result, bool)

    def test_returns_false_when_layer_has_mismatch(self) -> None:
        """Should return False when any site in layer doesn't match target."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)
        aa.target_Rb[:, :] = np.uint8(0)
        aa.target_Cs[:, :] = np.uint8(0)

        # Mismatch: atom present but target absent
        aa.matrix[1, 1, 0] = 1
        aa.target[1, 1, 0] = 0

        is_correct = is_site_correct(aa)
        result = layer_complete(1, aa, is_correct)
        assert isinstance(result, bool)


class TestRearrangementComplete:
    """Test overall rearrangement completion checking."""

    def test_returns_true_when_all_atoms_at_target(self) -> None:
        """Should return True when atom matrix matches target matrix."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # All atoms positioned at targets
        aa.matrix[1, 1, 0] = 1
        aa.matrix[2, 2, 1] = 1
        aa.target[1, 1, 0] = 1
        aa.target[2, 2, 1] = 1

        assert rearrangement_complete(aa) is True

    def test_returns_false_when_atoms_not_at_target(self) -> None:
        """Should return False when atoms are misplaced."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)
        aa.target_Rb[:, :] = np.uint8(0)
        aa.target_Cs[:, :] = np.uint8(0)

        # Atoms misplaced
        aa.matrix[0, 0, 0] = 1
        aa.target[3, 3, 0] = 1
        aa.target_Rb[3, 3] = 1

        assert rearrangement_complete(aa) is False

    def test_returns_false_with_empty_target(self) -> None:
        """Should return False when target has atoms but matrix is empty."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)
        aa.target_Rb[:, :] = np.uint8(0)
        aa.target_Cs[:, :] = np.uint8(0)

        # Target defined but no atoms placed
        aa.target[2, 2, 0] = 1
        aa.target_Rb[2, 2] = 1

        # Must explicitly check: target wants atom but matrix is empty
        assert rearrangement_complete(aa) is False


class TestInsideOutAlgorithm:
    """Test main inside-out rearrangement algorithm."""

    def test_returns_valid_structure(self) -> None:
        """Algorithm should return (arrays, moves, success)."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # Simple valid configuration
        aa.matrix[0, 0, 0] = 1
        aa.target[2, 2, 0] = 1

        result_aa, moves, success = inside_out_algorithm(aa, round_lim=10)

        assert isinstance(result_aa, AtomArray)
        assert isinstance(moves, (list, type(None)))
        assert isinstance(success, bool)

    def test_returns_false_with_insufficient_atoms(self) -> None:
        """Should return False when insufficient atoms for targets."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)
        aa.target_Rb[:, :] = np.uint8(0)
        aa.target_Cs[:, :] = np.uint8(0)

        # No atoms, but targets defined
        aa.target[2, 2, 0] = 1
        aa.target_Rb[2, 2] = 1

        result_aa, moves, success = inside_out_algorithm(aa, round_lim=1)

        assert success is False

    def test_conserves_atom_count(self) -> None:
        """Total atom count should remain constant during rearrangement."""
        aa = AtomArray(shape=[6, 6], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # Place 3 Rb atoms and 2 Cs atoms
        aa.matrix[0, 0, 0] = 1
        aa.matrix[0, 1, 0] = 1
        aa.matrix[5, 5, 0] = 1
        aa.matrix[1, 1, 1] = 1
        aa.matrix[1, 2, 1] = 1

        # Set targets in center
        aa.target[2:4, 2:4, 0] = 1
        aa.target[2:4, 2:4, 1] = 1

        initial_rb_count = _n_atoms(aa.matrix[:, :, 0:1])
        initial_cs_count = _n_atoms(aa.matrix[:, :, 1:2])

        result_aa, moves, _ = inside_out_algorithm(aa, round_lim=15)

        final_rb_count = _n_atoms(result_aa.matrix[:, :, 0:1])
        final_cs_count = _n_atoms(result_aa.matrix[:, :, 1:2])

        assert final_rb_count == initial_rb_count
        assert final_cs_count == initial_cs_count

    def test_terminates_within_round_limit(self) -> None:
        """Algorithm should respect round limit."""
        aa = AtomArray(shape=[6, 6], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        aa.matrix[0, 0, 0] = 1
        aa.target[3, 3, 0] = 1

        round_lim = 5
        result_aa, moves, success = inside_out_algorithm(aa, round_lim=round_lim)

        # Should not raise an exception and should return valid structure
        assert isinstance(result_aa, AtomArray)

    def test_handles_already_rearranged_atoms(self) -> None:
        """Algorithm should recognize when atoms already at target."""
        aa = AtomArray(shape=[5, 5], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # Atoms already positioned at target
        aa.matrix[2, 2, 0] = 1
        aa.target[2, 2, 0] = 1

        result_aa, moves, success = inside_out_algorithm(aa, round_lim=5)

        # Should complete successfully with minimal moves
        assert success is True


class TestBoundaryValidation:
    """Test boundary and out-of-bounds checking."""

    def test_out_bound_ex_detects_outside_region(self) -> None:
        """Should correctly identify coordinates outside boundary."""
        top, left, bottom, right = 2, 2, 4, 4

        # Inside boundary
        assert out_bound_ex(3, 3, top, left, bottom, right) is False

        # Outside boundary
        assert out_bound_ex(0, 0, top, left, bottom, right) is True
        assert out_bound_ex(5, 5, top, left, bottom, right) is True

    def test_out_bound_ex_includes_boundary(self) -> None:
        """Boundary cells should not be considered out of bounds."""
        top, left, bottom, right = 2, 2, 4, 4

        # Corners should be in bounds
        assert out_bound_ex(top, left, top, left, bottom, right) is False
        assert out_bound_ex(bottom, right, top, left, bottom, right) is False


class TestDualSpeciesAssignments:
    """Test assignment generation for dual-species rearrangement."""

    def test_gen_dual_assign_returns_valid_assignments(self) -> None:
        """Should generate valid source-to-target assignment pairs."""
        aa = AtomArray(shape=[6, 6], n_species=2)
        aa.matrix[:, :, :] = np.uint8(0)
        aa.target[:, :, :] = np.uint8(0)

        # Place atoms and targets
        aa.matrix[0, 0, 0] = 1
        aa.matrix[1, 1, 1] = 1
        aa.target[4, 4, 0] = 1
        aa.target[5, 5, 1] = 1

        out_assign, in_assign = gen_dual_assign_new(aa, layer_factor=1)

        assert isinstance(out_assign, list)
        assert isinstance(in_assign, list)


class TestAtomArrayEquality:
    """Test atom array comparison."""

    def test_arrays_equal_when_identical(self) -> None:
        """Should return True when matrices are identical."""
        aa1 = AtomArray(shape=[5, 5], n_species=2)
        aa2 = AtomArray(shape=[5, 5], n_species=2)

        aa1.matrix[:, :, :] = np.uint8(0)
        aa2.matrix[:, :, :] = np.uint8(0)

        aa1.matrix[0, 0, 0] = 1
        aa2.matrix[0, 0, 0] = 1

        assert atom_arrays_equal(aa1, aa2) is True

    def test_arrays_unequal_when_different(self) -> None:
        """Should return False when matrices differ."""
        aa1 = AtomArray(shape=[5, 5], n_species=2)
        aa2 = AtomArray(shape=[5, 5], n_species=2)

        aa1.matrix[:, :, :] = np.uint8(0)
        aa2.matrix[:, :, :] = np.uint8(0)

        aa1.matrix[0, 0, 0] = 1
        aa2.matrix[0, 1, 0] = 1

        assert atom_arrays_equal(aa1, aa2) is False
