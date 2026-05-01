import pytest
import numpy as np

from atommovr.utils.core import (
    Configurations,
    ArrayGeometry,
    PhysicalParams,
    CONFIGURATION_PLOT_LABELS,
    random_loading,
    generate_random_init_target_configs,
    generate_random_init_configs,
    generate_random_target_configs,
    count_atoms_in_columns,
    left_right_atom_in_row,
    top_bot_atom_in_col,
    find_lowest_atom_in_col,
    get_move_distance,
    atom_loss,
    atom_loss_dual,
    count_atoms_in_row,
    calculate_filling_fraction,
    save_frames,
    generate_middle_fifty,
)

###########################
# Test Enum Classes       #
###########################


class TestConfigurations:
    """Tests for the Configurations enum class."""

    def test_enum_values(self):
        """Test that all enum values have correct integer values."""
        assert Configurations.ZEBRA_HORIZONTAL == 0
        assert Configurations.ZEBRA_VERTICAL == 1
        assert Configurations.CHECKERBOARD == 2
        assert Configurations.MIDDLE_FILL == 3
        assert Configurations.Left_Sweep == 4
        assert Configurations.SEPARATE == 5
        assert Configurations.RANDOM == 6

    def test_enum_count(self):
        """Test that there are exactly 7 configurations."""
        assert len(Configurations) == 7

    def test_enum_is_int(self):
        """Test that enum values behave as integers."""
        assert isinstance(Configurations.CHECKERBOARD.value, int)
        assert Configurations.CHECKERBOARD + 1 == 3

    def test_enum_from_value(self):
        """Test lookup of enum by integer value."""
        assert Configurations(0) == Configurations.ZEBRA_HORIZONTAL
        assert Configurations(2) == Configurations.CHECKERBOARD

    def test_invalid_enum_value_raises(self):
        """Test that invalid enum values raise ValueError."""
        with pytest.raises(ValueError):
            Configurations(100)

    def test_configuration_plot_labels(self):
        """Test that CONFIGURATION_PLOT_LABELS has proper mappings."""
        assert (
            CONFIGURATION_PLOT_LABELS[Configurations.ZEBRA_HORIZONTAL]
            == "Horizontal zebra stripes"
        )
        assert CONFIGURATION_PLOT_LABELS[Configurations.CHECKERBOARD] == "Checkerboard"
        assert (
            CONFIGURATION_PLOT_LABELS[Configurations.MIDDLE_FILL]
            == "Middle fill rectangle"
        )
        assert CONFIGURATION_PLOT_LABELS[Configurations.RANDOM] == "Random"


class TestArrayGeometry:
    """Tests for the ArrayGeometry enum class."""

    def test_enum_values(self):
        """Test that all enum values have correct integer values."""
        assert ArrayGeometry.SQUARE == 0
        assert ArrayGeometry.RECTANGULAR == 1
        assert ArrayGeometry.TRIANGULAR == 2
        assert ArrayGeometry.BRAVAIS == 3
        assert ArrayGeometry.DECORATED_BRAVAIS == 4

    def test_enum_count(self):
        """Test that there are exactly 6 geometry types."""
        assert len(ArrayGeometry) == 6

    def test_enum_from_value(self):
        """Test lookup of enum by integer value."""
        assert ArrayGeometry(0) == ArrayGeometry.SQUARE
        assert ArrayGeometry(1) == ArrayGeometry.RECTANGULAR


###########################
# Test PhysicalParams     #
###########################


class TestPhysicalParams:
    """Tests for the PhysicalParams class."""

    def test_default_initialization(self):
        """Test default parameter values."""
        params = PhysicalParams()
        assert params.AOD_speed == 0.1
        assert params.spacing == 5e-6
        assert params.loading_prob == 0.6
        assert params.target_occup_prob == 0.5

    def test_custom_initialization(self):
        """Test custom parameter values."""
        params = PhysicalParams(
            AOD_speed=0.2, spacing=1e-5, loading_prob=0.8, target_occup_prob=0.7
        )
        assert params.AOD_speed == 0.2
        assert params.spacing == 1e-5
        assert params.loading_prob == 0.8
        assert params.target_occup_prob == 0.7

    def test_boundary_loading_prob_values(self):
        """Test boundary values (0 and 1) for loading_prob."""
        params_zero = PhysicalParams(loading_prob=0)
        assert params_zero.loading_prob == 0

        params_one = PhysicalParams(loading_prob=1)
        assert params_one.loading_prob == 1

    def test_boundary_target_occup_prob_values(self):
        """Test boundary values (0 and 1) for target_occup_prob."""
        params_zero = PhysicalParams(target_occup_prob=0)
        assert params_zero.target_occup_prob == 0

        params_one = PhysicalParams(target_occup_prob=1)
        assert params_one.target_occup_prob == 1

    def test_invalid_loading_prob_above_one(self):
        """Test that loading_prob > 1 raises ValueError."""
        with pytest.raises(ValueError, match="loading_prob"):
            PhysicalParams(loading_prob=1.1)

    def test_invalid_loading_prob_below_zero(self):
        """Test that loading_prob < 0 raises ValueError."""
        with pytest.raises(ValueError, match="loading_prob"):
            PhysicalParams(loading_prob=-0.1)

    def test_invalid_target_occup_prob_above_one(self):
        """Test that target_occup_prob > 1 raises ValueError."""
        with pytest.raises(ValueError, match="target_occup_prob"):
            PhysicalParams(target_occup_prob=1.5)

    def test_invalid_target_occup_prob_below_zero(self):
        """Test that target_occup_prob < 0 raises ValueError."""
        with pytest.raises(ValueError, match="target_occup_prob"):
            PhysicalParams(target_occup_prob=-0.5)


###########################
# Test Random Generation  #
###########################


class TestRandomLoading:
    """Tests for random_loading function."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        result = random_loading([5, 5], 0.5)
        assert result.shape == (5, 5)

    def test_rectangular_shape(self):
        """Test non-square array shapes."""
        result = random_loading([3, 7], 0.5)
        assert result.shape == (3, 7)

    def test_values_are_binary(self):
        """Test that all values are 0 or 1."""
        result = random_loading([10, 10], 0.5)
        assert np.all((result == 0) | (result == 1))

    def test_probability_zero(self):
        """Test that probability 0 produces empty array."""
        result = random_loading([5, 5], 0)
        assert np.sum(result) == 0

    def test_probability_one(self):
        """Test that probability 1 produces full array."""
        result = random_loading([5, 5], 1)
        assert np.sum(result) == 25

    def test_probability_distribution(self):
        """Test that average filling is approximately equal to probability."""
        # Use large array for statistical significance
        result = random_loading([100, 100], 0.6)
        filling = np.sum(result) / (100 * 100)
        assert 0.5 < filling < 0.7  # Allow for random variation


class TestGenerateRandomInitConfigs:
    """Tests for generate_random_init_configs function."""

    def test_single_species_output_count(self):
        """Test that correct number of configs are generated."""
        configs = generate_random_init_configs(5, 0.6, 10, n_species=1)
        assert len(configs) == 5

    def test_single_species_shape(self):
        """Test single species config shape."""
        configs = generate_random_init_configs(3, 0.6, 8, n_species=1)
        assert all(c.shape == (8, 8) for c in configs)

    def test_dual_species_shape(self):
        """Test dual species config shape (3D array)."""
        configs = generate_random_init_configs(3, 0.6, 8, n_species=2)
        assert all(c.shape == (8, 8, 2) for c in configs)

    def test_dual_species_no_overlap(self):
        """Test that dual species configs have no overlapping atoms."""
        configs = generate_random_init_configs(10, 0.6, 10, n_species=2)
        for config in configs:
            # At each (i,j), at most one species can be present
            for i in range(10):
                for j in range(10):
                    assert not (config[i, j, 0] == 1 and config[i, j, 1] == 1)

    def test_invalid_n_species(self):
        """Test that invalid n_species raises ValueError."""
        with pytest.raises(ValueError):
            generate_random_init_configs(5, 0.6, 10, n_species=3)

    def test_binary_values_single_species(self):
        """Test that all values are 0 or 1 for single species."""
        configs = generate_random_init_configs(3, 0.6, 10, n_species=1)
        for config in configs:
            assert np.all((config == 0) | (config == 1))

    def test_binary_values_dual_species(self):
        """Test that all values are 0 or 1 for dual species."""
        configs = generate_random_init_configs(3, 0.6, 10, n_species=2)
        for config in configs:
            assert np.all((config == 0) | (config == 1))


class TestGenerateRandomTargetConfigs:
    """Tests for generate_random_target_configs function."""

    def test_output_count(self):
        """Test that correct number of configs are generated."""
        configs = generate_random_target_configs(10, 0.5, [5, 5])
        assert len(configs) == 10

    def test_output_shape(self):
        """Test output array shapes."""
        configs = generate_random_target_configs(3, 0.5, [7, 9])
        assert all(c.shape == (7, 9) for c in configs)

    def test_binary_values(self):
        """Test that all values are 0 or 1."""
        configs = generate_random_target_configs(5, 0.5, [10, 10])
        for config in configs:
            assert np.all((config == 0) | (config == 1))

    def test_zero_probability(self):
        """Test that probability 0 produces empty configs."""
        configs = generate_random_target_configs(3, 0, [5, 5])
        for config in configs:
            assert np.sum(config) == 0

    def test_full_probability(self):
        """Test that probability 1 produces full configs."""
        configs = generate_random_target_configs(3, 1, [5, 5])
        for config in configs:
            assert np.sum(config) == 25


class TestGenerateRandomInitTargetConfigs:
    """Tests for generate_random_init_target_configs function."""

    def test_output_lengths(self):
        """Test that correct number of init and target configs are generated."""
        init_configs, target_configs = generate_random_init_target_configs(5, 0.6, 10)
        assert len(init_configs) == 5

    def test_init_config_shape(self):
        """Test initial config shapes."""
        init_configs, _ = generate_random_init_target_configs(3, 0.6, 8)
        assert all(c.shape == (8, 8) for c in init_configs)

    def test_with_random_target(self):
        """Test generating random target configurations."""
        init_configs, target_configs = generate_random_init_target_configs(
            5, 0.6, 10, target_config=[Configurations.RANDOM]
        )
        assert len(init_configs) == 5
        assert len(target_configs) == 5


###########################
# Test Atom Counting      #
###########################


class TestCountAtomsInColumns:
    """Tests for count_atoms_in_columns function."""

    def test_basic_counting(self):
        """Test basic column counting."""
        matrix = [[1, 0, 1], [0, 1, 0], [1, 1, 0]]
        result = count_atoms_in_columns(matrix)
        assert result == [2, 2, 1]

    def test_empty_matrix(self):
        """Test with empty (all zeros) matrix."""
        matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        result = count_atoms_in_columns(matrix)
        assert result == [0, 0, 0]

    def test_full_matrix(self):
        """Test with full (all ones) matrix."""
        matrix = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        result = count_atoms_in_columns(matrix)
        assert result == [3, 3, 3]

    def test_single_column(self):
        """Test with single column matrix."""
        matrix = [[1], [0], [1], [1]]
        result = count_atoms_in_columns(matrix)
        assert result == [3]

    def test_single_row(self):
        """Test with single row matrix."""
        matrix = [[1, 0, 1, 0, 1]]
        result = count_atoms_in_columns(matrix)
        assert result == [1, 0, 1, 0, 1]

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        matrix = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        result = count_atoms_in_columns(matrix)
        assert result == [2, 2, 1]


class TestCountAtomsInRow:
    """Tests for count_atoms_in_row function."""

    def test_basic_counting(self):
        """Test basic row counting."""
        row = [1, 0, 1, 1, 0]
        assert count_atoms_in_row(row) == 3

    def test_empty_row(self):
        """Test empty row."""
        row = [0, 0, 0, 0, 0]
        assert count_atoms_in_row(row) == 0

    def test_full_row(self):
        """Test full row."""
        row = [1, 1, 1, 1, 1]
        assert count_atoms_in_row(row) == 5

    def test_single_element(self):
        """Test single element row."""
        assert count_atoms_in_row([1]) == 1
        assert count_atoms_in_row([0]) == 0

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        row = np.array([1, 0, 1, 1, 0])
        assert count_atoms_in_row(row) == 3


###########################
# Test Atom Locating      #
###########################


class TestLeftRightAtomInRow:
    """Tests for left_right_atom_in_row function.

    Note: direction=1 finds rightmost atom (reverse scan),
          direction=-1 finds leftmost atom (forward scan).
    """

    def test_find_rightmost(self):
        """Test finding rightmost atom (direction=1)."""
        row = [0, 1, 0, 1, 0]
        assert left_right_atom_in_row(row, 1) == 3

    def test_find_leftmost(self):
        """Test finding leftmost atom (direction=-1)."""
        row = [0, 1, 0, 1, 0]
        assert left_right_atom_in_row(row, -1) == 1

    def test_single_atom(self):
        """Test with single atom in row."""
        row = [0, 0, 1, 0, 0]
        assert left_right_atom_in_row(row, 1) == 2
        assert left_right_atom_in_row(row, -1) == 2

    def test_atom_at_edge_left(self):
        """Test with atom at left edge."""
        row = [1, 0, 0, 0, 0]
        assert left_right_atom_in_row(row, -1) == 0

    def test_atom_at_edge_right(self):
        """Test with atom at right edge."""
        row = [0, 0, 0, 0, 1]
        assert left_right_atom_in_row(row, 1) == 4

    def test_empty_row(self):
        """Test with empty row returns None."""
        row = [0, 0, 0, 0, 0]
        assert left_right_atom_in_row(row, 1) is None
        assert left_right_atom_in_row(row, -1) is None

    def test_full_row(self):
        """Test with full row."""
        row = [1, 1, 1, 1, 1]
        assert left_right_atom_in_row(row, 1) == 4  # rightmost
        assert left_right_atom_in_row(row, -1) == 0  # leftmost


class TestTopBotAtomInCol:
    """Tests for top_bot_atom_in_col function.

    Note: direction=1 finds bottommost atom (reverse scan),
          direction=-1 finds topmost atom (forward scan).
    """

    def test_find_bottommost(self):
        """Test finding bottommost atom (direction=1)."""
        col = [0, 1, 0, 1, 0]
        assert top_bot_atom_in_col(col, 1) == 3

    def test_find_topmost(self):
        """Test finding topmost atom (direction=-1)."""
        col = [0, 1, 0, 1, 0]
        assert top_bot_atom_in_col(col, -1) == 1

    def test_single_atom(self):
        """Test with single atom in column."""
        col = [0, 0, 1, 0, 0]
        assert top_bot_atom_in_col(col, 1) == 2
        assert top_bot_atom_in_col(col, -1) == 2

    def test_empty_column(self):
        """Test with empty column returns None."""
        col = [0, 0, 0, 0, 0]
        assert top_bot_atom_in_col(col, 1) is None
        assert top_bot_atom_in_col(col, -1) is None


class TestFindLowestAtomInCol:
    """Tests for find_lowest_atom_in_col function."""

    def test_basic_case(self):
        """Test finding lowest atom."""
        col = [0, 1, 0, 1, 0]
        assert find_lowest_atom_in_col(col) == 3

    def test_single_atom_top(self):
        """Test with single atom at top."""
        col = [1, 0, 0, 0, 0]
        assert find_lowest_atom_in_col(col) == 0

    def test_single_atom_bottom(self):
        """Test with single atom at bottom."""
        col = [0, 0, 0, 0, 1]
        assert find_lowest_atom_in_col(col) == 4

    def test_empty_column(self):
        """Test with empty column returns None."""
        col = [0, 0, 0, 0, 0]
        assert find_lowest_atom_in_col(col) is None

    def test_full_column(self):
        """Test with full column returns last index."""
        col = [1, 1, 1, 1, 1]
        assert find_lowest_atom_in_col(col) == 4


###########################
# Test Movement           #
###########################


class TestGetMoveDistance:
    """Tests for get_move_distance function."""

    def test_basic_distance(self):
        """Test basic Manhattan distance calculation."""
        distance = get_move_distance(0, 0, 2, 3, spacing=5e-6)
        expected = (2 + 3) * 5e-6  # = 2.5e-5
        assert distance == expected

    def test_horizontal_only(self):
        """Test horizontal-only movement."""
        distance = get_move_distance(1, 1, 1, 4, spacing=1e-6)
        expected = 3 * 1e-6
        assert distance == expected

    def test_vertical_only(self):
        """Test vertical-only movement."""
        distance = get_move_distance(0, 2, 5, 2, spacing=1e-6)
        expected = 5 * 1e-6
        assert distance == expected

    def test_zero_distance(self):
        """Test same position (zero distance)."""
        distance = get_move_distance(3, 3, 3, 3, spacing=5e-6)
        assert distance == 0

    def test_negative_coordinates(self):
        """Test that negative direction is handled correctly (via abs)."""
        distance = get_move_distance(5, 5, 2, 3, spacing=5e-6)
        expected = (3 + 2) * 5e-6
        assert distance == expected

    def test_default_spacing(self):
        """Test with default spacing value."""
        distance = get_move_distance(0, 0, 1, 1)
        expected = 2 * 5e-6
        assert distance == expected

    def test_custom_spacing(self):
        """Test with custom spacing value."""
        distance = get_move_distance(0, 0, 1, 1, spacing=1e-5)
        expected = 2 * 1e-5
        assert distance == expected


###########################
# Test Atom Loss          #
###########################


class TestAtomLoss:
    """Tests for atom_loss function."""

    def test_output_shape_preserved(self):
        """Test that output shape matches input shape."""
        matrix = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
        result, _ = atom_loss(matrix, move_time=0.1, lifetime=30)
        assert result.shape == matrix.shape

    def test_zero_move_time_no_loss(self):
        """Test that zero move time results in no loss (prob=1)."""
        matrix = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        result, flag = atom_loss(matrix, move_time=0, lifetime=30)
        # With move_time=0, exp(-0/30) = 1, so all atoms should survive
        assert np.array_equal(result, matrix)
        assert flag == 0

    def test_very_long_time_significant_loss(self):
        """Test that very long move time results in significant loss."""
        matrix = np.ones((20, 20))
        result, _ = atom_loss(matrix, move_time=100, lifetime=1)
        # With move_time=100, lifetime=1, exp(-100) ≈ 0, most atoms lost
        assert np.sum(result) < np.sum(matrix) * 0.1

    def test_loss_flag_when_loss_occurs(self):
        """Test that loss flag is set when atoms are lost."""
        np.random.seed(42)  # For reproducibility
        matrix = np.ones((10, 10))
        # With very long time, loss should definitely occur
        _, flag = atom_loss(matrix, move_time=1000, lifetime=1)
        assert flag == 1

    def test_empty_matrix(self):
        """Test with empty matrix."""
        matrix = np.zeros((5, 5))
        result, flag = atom_loss(matrix, move_time=10, lifetime=30)
        assert np.sum(result) == 0
        assert flag == 0

    def test_values_remain_binary(self):
        """Test that result contains only 0s and 1s."""
        matrix = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
        result, _ = atom_loss(matrix, move_time=0.5, lifetime=30)
        assert np.all((result == 0) | (result == 1))


class TestAtomLossDual:
    """Tests for atom_loss_dual function."""

    def test_output_shape_preserved(self):
        """Test that output shape matches input shape."""
        matrix = np.zeros((5, 5, 2))
        matrix[:, :, 0] = random_loading([5, 5], 0.3)
        matrix[:, :, 1] = random_loading([5, 5], 0.3)
        result, _ = atom_loss_dual(matrix, move_time=0.1, lifetime=30)
        assert result.shape == matrix.shape

    def test_empty_matrix(self):
        """Test with empty dual-species matrix."""
        matrix = np.zeros((5, 5, 2))
        result, flag = atom_loss_dual(matrix, move_time=10, lifetime=30)
        assert np.sum(result) == 0
        assert flag == 0  # No loss can occur if no atoms

    def test_zero_move_time_no_loss(self):
        """Test that zero move time results in no loss."""
        matrix = np.ones((5, 5, 2))
        result, flag = atom_loss_dual(matrix, move_time=0, lifetime=30)
        assert np.array_equal(result, matrix)
        assert flag == 0

    def test_values_remain_binary(self):
        """Test that result contains only 0s and 1s."""
        matrix = np.zeros((5, 5, 2))
        matrix[:, :, 0] = random_loading([5, 5], 0.5)
        matrix[:, :, 1] = random_loading([5, 5], 0.5)
        result, _ = atom_loss_dual(matrix, move_time=1.0, lifetime=30)
        assert np.all((result == 0) | (result == 1))

    def test_very_long_time_significant_loss(self):
        """Test that very long move time results in significant loss."""
        matrix = np.ones((10, 10, 2))
        result, flag = atom_loss_dual(matrix, move_time=100, lifetime=1)
        # With move_time=100, lifetime=1, exp(-100) ≈ 0, most atoms lost
        assert np.sum(result) < np.sum(matrix) * 0.1
        assert flag == 1

    def test_both_species_affected_equally(self):
        """Test that both species at same site get same loss mask."""
        # Create matrix where both species occupy all sites
        matrix = np.ones((10, 10, 2))
        result, _ = atom_loss_dual(matrix, move_time=5, lifetime=10)
        # At each site, both species should have same survival status
        for i in range(10):
            for j in range(10):
                assert result[i, j, 0] == result[i, j, 1]


###########################
# Test Utility Functions  #
###########################


class TestCalculateFillingFraction:
    """Tests for calculate_filling_fraction function."""

    def test_basic_calculation(self):
        """Test basic filling fraction calculation."""
        assert calculate_filling_fraction(3, 5) == 60.0

    def test_zero_atoms(self):
        """Test with zero atoms."""
        assert calculate_filling_fraction(0, 10) == 0.0

    def test_full_row(self):
        """Test with full row."""
        assert calculate_filling_fraction(10, 10) == 100.0

    def test_half_filled(self):
        """Test with half filled."""
        assert calculate_filling_fraction(5, 10) == 50.0

    def test_small_fraction(self):
        """Test small fraction."""
        assert calculate_filling_fraction(2, 10) == 20.0

    def test_large_numbers(self):
        """Test with large numbers."""
        assert calculate_filling_fraction(1000, 10000) == 10.0


class TestSaveFrames:
    """Tests for save_frames function."""

    def test_basic_save(self):
        """Test basic frame saving."""
        temp = [1, 2, 3]
        combined = []
        temp_new, combined_new = save_frames(temp, combined)
        assert combined_new == [1, 2, 3]
        assert temp_new == []

    def test_extending_combined(self):
        """Test extending existing combined frames."""
        temp = [4, 5, 6]
        combined = [1, 2, 3]
        temp_new, combined_new = save_frames(temp, combined)
        assert combined_new == [1, 2, 3, 4, 5, 6]
        assert temp_new == []

    def test_empty_temp(self):
        """Test with empty temp frames."""
        temp = []
        combined = [1, 2, 3]
        temp_new, combined_new = save_frames(temp, combined)
        assert combined_new == [1, 2, 3]
        assert temp_new == []

    def test_reference_behavior(self):
        """Test that original lists are modified in place."""
        temp = [1, 2, 3]
        combined = []
        save_frames(temp, combined)
        # Original lists should be modified
        assert temp == []
        assert combined == [1, 2, 3]


class TestGenerateMiddleFifty:
    """Tests for generate_middle_fifty function."""

    def test_default_threshold(self):
        """Test with default filling threshold."""
        result = generate_middle_fifty(10)
        # max_L^2 / 100 < 0.5 => max_L < 7.07 => max_L = 7
        assert result == [7, 7]

    def test_custom_threshold_half(self):
        """Test with 0.5 threshold."""
        result = generate_middle_fifty(10, 0.5)
        assert result == [7, 7]

    def test_custom_threshold_quarter(self):
        """Test with 0.25 threshold."""
        result = generate_middle_fifty(20, 0.25)
        # max_L^2 / 400 < 0.25 => max_L < 10 => but we also need sqrt(100) = 10
        # Let's verify the algorithm
        # Starting at 20, while (max_L^2)/400 >= 0.25: decrement
        # 400/400 = 1.0 >= 0.25, max_L=19
        # 361/400 = 0.9025 >= 0.25, max_L=18
        # ... continues until less than 0.25
        # We need max_L^2 < 100, so max_L < 10
        # So final max_L = 9
        assert result == [9, 9]

    def test_large_array(self):
        """Test with larger array."""
        result = generate_middle_fifty(100, 0.5)
        # sqrt(5000) ≈ 70.7, so max_L = 70
        assert result == [70, 70]

    def test_small_array(self):
        """Test with small array."""
        result = generate_middle_fifty(5, 0.5)
        # sqrt(12.5) ≈ 3.5, so max_L = 3
        assert result == [3, 3]

    def test_high_threshold(self):
        """Test with high filling threshold."""
        result = generate_middle_fifty(10, 0.9)
        # sqrt(90) ≈ 9.5, so max_L = 9
        assert result == [9, 9]

    def test_low_threshold(self):
        """Test with low filling threshold."""
        result = generate_middle_fifty(10, 0.1)
        # sqrt(10) ≈ 3.16, so max_L = 3
        assert result == [3, 3]

    def test_returns_list(self):
        """Test that result is a list."""
        result = generate_middle_fifty(10)
        assert isinstance(result, list)
        assert len(result) == 2
