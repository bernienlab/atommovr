import numpy as np
import pytest

from atommovr.utils.core import Configurations, array_shape_for_geometry
from atommovr.utils import make_single_species_gif
from atommovr.utils.AtomArray import AtomArray
from atommovr.algorithms.single_species import (
    Hungarian,
    ParallelHungarian,
    ParallelLBAP,
    BCv2,
    PCFA,
    Tetris,
    BalanceAndCompact,
    GeneralizedBalance,
)
from atommovr.algorithms.dual_species import InsideOut, NaiveParHung
from atommovr.algorithms.source.Hungarian_works import (
    parallel_LBAP_algorithm_works,
    parallel_Hungarian_algorithm_works,
)
from atommovr.algorithms.source.scaling_lower_bound import (
    calculate_Zstar_better,
    calculate_Zstar,
    calculate_LB,
    get_Zstar_lower_bound,
)
from atommovr.utils.imaging.visualization import visualize_move_batches
from atommovr.utils.errormodels import (
    ZeroNoise,
    UniformVacuumTweezerError,
    YbRydbergAODErrorModel,
)
import random

ERROR_MODELS = [ZeroNoise, UniformVacuumTweezerError, YbRydbergAODErrorModel]


def _default_source_state(array_shape: tuple[int, int], target_size: int) -> np.ndarray:
    rows, cols = array_shape
    t = target_size
    state = np.zeros(array_shape, dtype=int)
    band_rows = min(rows, t + 2)
    band_cols = min(cols, t + 2)
    state[:band_rows, :] = 1
    state[:, :band_cols] = 1
    state[:, -band_cols:] = 1
    return state


def _pcfa_source_state(array_shape: tuple[int, int], target_size: int) -> np.ndarray:
    rows, cols = array_shape
    state = np.zeros(array_shape, dtype=int)
    right_band = max(2, target_size // 2)
    top_band = min(rows, target_size + 2)
    state[:top_band, :] = 1
    state[:, cols - right_band :] = 1
    return state


ALGORITHM_CASES = [
    {
        "name": "Hungarian",
        "cls": Hungarian,
        "target_size": 4,
        "initializer": _default_source_state,
        "kwargs": {"do_ejection": False},
    },
    {
        "name": "ParallelHungarian",
        "cls": ParallelHungarian,
        "target_size": 4,
        "initializer": _default_source_state,
        "kwargs": {"do_ejection": False},
    },
    {
        "name": "ParallelLBAP",
        "cls": ParallelLBAP,
        "target_size": 4,
        "initializer": _default_source_state,
        "kwargs": {"do_ejection": False},
    },
    {
        "name": "BalanceAndCompact",
        "cls": BalanceAndCompact,
        "target_size": 4,
        "initializer": _default_source_state,
        "kwargs": {"do_ejection": False},
    },
    {
        "name": "GeneralizedBalance",
        "cls": GeneralizedBalance,
        "target_size": 4,
        "initializer": _default_source_state,
        "kwargs": {"do_ejection": False},
    },
    {
        "name": "BCv2",
        "cls": BCv2,
        "target_size": 4,
        "initializer": _default_source_state,
        "kwargs": {"do_ejection": False},
    },
    {
        "name": "PCFA",
        "cls": PCFA,
        "target_size": 4,
        "initializer": _pcfa_source_state,
        "kwargs": {"do_ejection": False},
    },
    {
        "name": "Tetris",
        "cls": Tetris,
        "target_size": 4,
        "initializer": _default_source_state,
        "kwargs": {"do_ejection": False},
    },
]


def _centered_target_mask(
    array_shape: tuple[int, int], target_size: int
) -> tuple[np.ndarray, tuple[int, int]]:
    mask = np.zeros(array_shape, dtype=int)
    r0 = max(0, (array_shape[0] - target_size) // 2)
    c0 = max(0, (array_shape[1] - target_size) // 2)
    mask[r0 : r0 + target_size, c0 : c0 + target_size] = 1
    return mask, (r0, c0)


def _build_array(
    array_shape: tuple[int, int], target_size: int, initializer
) -> tuple[AtomArray, np.ndarray, tuple[int, int]]:
    mask, origin = _centered_target_mask(array_shape, target_size)
    state = initializer(array_shape, target_size)
    state[mask == 1] = 0

    arr = AtomArray(list(array_shape), n_species=1)
    arr.matrix[:, :, 0] = state
    arr.target = mask.reshape(array_shape[0], array_shape[1], 1)
    return arr, mask, origin


def _contains_target_block(state: np.ndarray, target_size: int) -> bool:
    rows, cols = state.shape
    t = target_size
    if t > rows or t > cols:
        return False
    block = np.ones((t, t), dtype=int)
    for r0 in range(rows - t + 1):
        for c0 in range(cols - t + 1):
            if np.array_equal(state[r0 : r0 + t, c0 : c0 + t], block):
                return True
    return False


def _line_shift_state(num_cols: int, fill: int) -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros((1, num_cols), dtype=int)
    state[0, :fill] = 1
    target = np.zeros_like(state)
    target[0, num_cols - fill :] = 1
    return state, target


def _load_until_sufficient(arr: AtomArray, max_tries: int = 50) -> bool:
    """Reload tweezers until there are enough atoms for the target."""
    for _ in range(max_tries):
        arr.load_tweezers()
        if np.sum(arr.matrix) >= np.sum(arr.target):
            return True
    return False


def _dual_species_target_filled(arr: AtomArray) -> bool:
    """Check that every target site is filled for each species."""
    for s in range(arr.n_species):
        target_plane = arr.target[:, :, s]
        matrix_plane = arr.matrix[:, :, s]
        if not np.all(matrix_plane[target_plane == 1] == 1):
            return False
    return True


def _load_until_sufficient_dual(arr: AtomArray, max_tries: int = 100) -> bool:
    """Reload tweezers until each species has enough atoms for its target."""
    for _ in range(max_tries):
        arr.load_tweezers()
        sufficient = all(
            np.sum(arr.matrix[:, :, s]) >= np.sum(arr.target[:, :, s])
            for s in range(arr.n_species)
        )
        if sufficient:
            return True
    return False


@pytest.mark.parametrize("case", ALGORITHM_CASES, ids=lambda case: case["name"])
def test_single_species_algorithms_cover_target_shapes(case):
    target_size = case["target_size"]
    initializer = case["initializer"]
    algo = case["cls"]()

    # compute array shape using central geometry helper
    array_shape = tuple(
        array_shape_for_geometry(
            getattr(algo, "preferred_geometry_spec", None), target_size
        )
    )

    arr, mask, (r0, c0) = _build_array(array_shape, target_size, initializer)
    _, move_batches, success = algo.get_moves(arr, **case.get("kwargs", {}))
    assert success, f"{case['name']} reported failure"

    # visualize_move_batches(arr, move_batches, save_path=None, title_suffix=f"{case['name']} Move Plan")
    # visualize_batch_moves_on_image(arr, move_batches, save_path=None, title_suffix=f"{case['name']} Move Plan Overlay")

    arr.evaluate_moves(move_batches)
    submatrix = arr.matrix[r0 : r0 + target_size, c0 : c0 + target_size, 0]
    assert np.array_equal(
        submatrix, np.ones((target_size, target_size), dtype=int)
    ), f"{case['name']} did not fill the target region"


@pytest.mark.parametrize("case", ALGORITHM_CASES, ids=lambda case: case["name"])
def test_single_species_algorithms_natively(case):
    target_size = case["target_size"]
    algo = case["cls"]()

    # make randomness deterministic for test reproducibility
    random.seed(42)
    np.random.seed(42)

    # choose array shape from geometry helper
    array_shape = tuple(
        array_shape_for_geometry(
            getattr(algo, "preferred_geometry_spec", None), target_size
        )
    )

    arr = AtomArray(list(array_shape), n_species=1)
    arr.generate_target(
        Configurations.MIDDLE_FILL,
        middle_size=(target_size, target_size),
        occupation_prob=0.6,
    )
    assert _load_until_sufficient(arr), "Could not load sufficient atoms for target"

    _, move_batches, success = algo.get_moves(arr, **case.get("kwargs", {}))

    visualize_move_batches(
        arr, move_batches, save_path=None, title_suffix=f"{case['name']} Move Plan"
    )
    # visualize_batch_moves_on_image(arr, move_batches, save_path=None, title_suffix=f"{case['name']} Move Plan Overlay")

    assert success, f"{case['name']} reported failure"

    arr.evaluate_moves(move_batches)
    filled = _contains_target_block(arr.matrix[:, :, 0], target_size)
    assert (
        filled
    ), f"{case['name']} did not realize the required {target_size} block anywhere in the array"


@pytest.mark.parametrize("case", ALGORITHM_CASES, ids=lambda case: case["name"])
def test_single_species_multiple_shots_natively(case):
    target_size = case["target_size"]
    algo = case["cls"]()

    # make randomness deterministic for test reproducibility
    random.seed(0)
    np.random.seed(0)

    array_shape = tuple(
        array_shape_for_geometry(
            getattr(algo, "preferred_geometry_spec", None), target_size
        )
    )
    arr = AtomArray(list(array_shape), n_species=1)

    n_shots = case.get("n_shots", 1)
    for shot in range(n_shots):
        arr.generate_target(
            Configurations.MIDDLE_FILL,
            middle_size=(target_size, target_size),
            occupation_prob=0.6,
        )
        assert _load_until_sufficient(
            arr
        ), f"Could not load sufficient atoms for target on shot {shot}"
        _, move_batches, success = algo.get_moves(arr, **case.get("kwargs", {}))

        visualize_move_batches(
            arr,
            move_batches,
            save_path=None,
            title_suffix=f"{case['name']}_{shot}_Move_Plan",
        )
        # visualize_batch_moves_on_image(arr, move_batches, save_path=None, title_suffix=f"{case['name']}_{shot}_Move_Plan_Overlay")

        assert success, f"{case['name']} reported failure on shot {shot}"

        arr.evaluate_moves(move_batches)
        filled = _contains_target_block(arr.matrix[:, :, 0], target_size)
        assert (
            filled
        ), f"{case['name']} did not realize the required {target_size} block anywhere in the array on shot {shot}"

        make_single_species_gif(
            arr, move_batches, savename=f"test_{case['name']}_shot{shot}_rearrangement"
        )


# def test_dual_species_algorithms_natively():
# 	"""Verify that dual-species algorithms run without errors.

# 	InsideOut is reliable and must succeed.  NaiveParHung is
# 	stochastic, so we only verify it does not crash and that a
# 	successful result is correct.
# 	"""
# 	target_size = 6
# 	algos = [InsideOut(), NaiveParHung()]

# 	for algo in algos:
# 		random.seed(42)
# 		np.random.seed(42)

# 		array_shape = tuple(array_shape_for_geometry(getattr(algo, "preferred_geometry_spec", None), target_size))
# 		arr = AtomArray(list(array_shape), n_species=2)
# 		arr.generate_target(Configurations.CHECKERBOARD, middle_size=(target_size, target_size), occupation_prob=0.6)
# 		assert _load_until_sufficient_dual(arr), f"Could not load sufficient atoms for target for {algo.__class__.__name__}"

# 		_, move_batches, success = algo.get_moves(arr, do_ejection=False)

# 		if isinstance(algo, InsideOut):
# 			assert success, f"{algo.__class__.__name__} reported failure"

# 		if success:
# 			arr.evaluate_moves(move_batches)
# 			assert _dual_species_target_filled(arr), (
# 				f"{algo.__class__.__name__} did not fill the target for both species")


def test_parallel_assignment_algorithms_complete_long_paths():
    state, target = _line_shift_state(12, 6)
    final_lbap, _, lbap_success = parallel_LBAP_algorithm_works(
        state.copy(), target.copy(), round_lim=50
    )
    assert (
        lbap_success
    ), "Parallel LBAP failed to complete deterministic long-path assignment"
    assert np.array_equal(final_lbap, target)

    final_hung, _, hung_success = parallel_Hungarian_algorithm_works(
        state.copy(), target.copy(), round_lim=50
    )
    assert (
        hung_success
    ), "Parallel Hungarian failed to complete deterministic long-path assignment"
    assert np.array_equal(final_hung, target)


@pytest.mark.parametrize("error_model_cls", ERROR_MODELS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("case", ALGORITHM_CASES, ids=lambda case: case["name"])
def test_algorithms_with_error_models(case, error_model_cls):
    """Run each algorithm with each error model to ensure behavior and stability.

    - For `ZeroNoise` we expect success and filling of the target region.
    - For other models we only assert that evaluation runs without exceptions and
      that the array shape/dtype are preserved.
    """
    # make randomness deterministic for test reproducibility
    random.seed(0)
    np.random.seed(0)

    target_size = case["target_size"]
    algo = case["cls"]()

    array_shape = tuple(
        array_shape_for_geometry(
            getattr(algo, "preferred_geometry_spec", None), target_size
        )
    )

    arr = AtomArray(list(array_shape), n_species=1)
    arr.generate_target(
        Configurations.MIDDLE_FILL,
        middle_size=(target_size, target_size),
        occupation_prob=0.6,
    )
    assert _load_until_sufficient(arr), "Could not load sufficient atoms for target"

    # attach the error model instance
    err = error_model_cls()
    arr.error_model = err

    _, move_batches, success = algo.get_moves(arr, **case.get("kwargs", {}))

    # visualize the planned moves
    # visualize_move_batches(arr, move_batches, save_path=None, title_suffix=f"{case['name']}_{error_model_cls.__name__}_Move_Plan")
    # visualize_batch_moves_on_image(arr, move_batches, save_path=None, title_suffix=f"{case['name']}_{error_model_cls.__name__}_Move_Plan_Overlay")

    # evaluation should not raise
    try:
        arr.evaluate_moves(move_batches)
    except Exception as e:
        pytest.fail(
            f"Evaluation raised with {error_model_cls.__name__} on {case['name']}: {e}"
        )

    # basic sanity checks
    assert isinstance(arr.matrix, np.ndarray)
    assert (
        arr.matrix.shape[0] == array_shape[0] and arr.matrix.shape[1] == array_shape[1]
    )

    target = arr.get_target()[:, :, 0]

    # For zero-noise we expect the algorithm to succeed and fill the target.
    if error_model_cls is ZeroNoise:
        assert success, f"{case['name']} reported failure with ZeroNoise"
        submatrix = arr.matrix[(target == 1)]
        assert np.all(
            submatrix == 1
        ), f"{case['name']} did not fill the target region with ZeroNoise"


# Parametrized dual-species tests

DUAL_SPECIES_CASES = [
    {"name": "InsideOut", "cls": InsideOut, "target_size": 6},
    {"name": "NaiveParHung", "cls": NaiveParHung, "target_size": 6},
]


# @pytest.mark.parametrize("case", DUAL_SPECIES_CASES, ids=lambda c: c["name"])
# def test_dual_species_cover_target(case):
# 	"""Dual-species algorithms must not crash; when they report success
# 	the target must actually be filled."""
# 	target_size = case["target_size"]
# 	algo = case["cls"]()

# 	random.seed(42)
# 	np.random.seed(42)

# 	array_shape = tuple(array_shape_for_geometry(
# 		getattr(algo, "preferred_geometry_spec", None), target_size))
# 	arr = AtomArray(list(array_shape), n_species=2)
# 	arr.generate_target(
# 		Configurations.CHECKERBOARD,
# 		middle_size=(target_size, target_size),
# 		occupation_prob=0.6,
# 	)
# 	assert _load_until_sufficient_dual(arr), (
# 		f"Could not load sufficient atoms for {case['name']}")

# 	_, move_batches, success = algo.get_moves(arr, do_ejection=False)

# 	# InsideOut is reliable; NaiveParHung is stochastic
# 	if isinstance(algo, InsideOut):
# 		assert success, f"{case['name']} reported failure"

# 	if success:
# 		arr.evaluate_moves(move_batches)
# 		assert _dual_species_target_filled(arr), (
# 			f"{case['name']} did not fill the dual-species target")


# @pytest.mark.parametrize("case", DUAL_SPECIES_CASES, ids=lambda c: c["name"])
# def test_dual_species_multiple_shots(case):
# 	"""Run multiple shots; at least one must succeed for InsideOut."""
# 	target_size = case["target_size"]
# 	algo = case["cls"]()

# 	random.seed(42)
# 	np.random.seed(42)

# 	array_shape = tuple(array_shape_for_geometry(
# 		getattr(algo, "preferred_geometry_spec", None), target_size))
# 	arr = AtomArray(list(array_shape), n_species=2)

# 	n_shots = 3
# 	any_success = False
# 	for shot in range(n_shots):
# 		arr.generate_target(
# 			Configurations.CHECKERBOARD,
# 			middle_size=(target_size, target_size),
# 			occupation_prob=0.6,
# 		)
# 		assert _load_until_sufficient_dual(arr), (
# 			f"Could not load sufficient atoms on shot {shot} for {case['name']}")

# 		_, move_batches, success = algo.get_moves(arr, do_ejection=False)

# 		if success:
# 			any_success = True
# 			arr.evaluate_moves(move_batches)
# 			assert _dual_species_target_filled(arr), (
# 				f"{case['name']} did not fill the dual-species target on shot {shot}")

# 	if isinstance(algo, InsideOut):
# 		assert any_success, f"InsideOut did not succeed in any of {n_shots} shots"


# @pytest.mark.parametrize("case", DUAL_SPECIES_CASES, ids=lambda c: c["name"])
# def test_dual_species_rejects_single_species(case):
# 	"""Dual-species algorithms require n_species=2."""
# 	algo = case["cls"]()
# 	arr = AtomArray([8, 8], n_species=1)
# 	arr.generate_target(Configurations.MIDDLE_FILL, middle_size=(4, 4))
# 	arr.load_tweezers()

# 	# InsideOut accesses matrix[:,:,1] which fails for 1-species arrays.
# 	with pytest.raises((IndexError, ValueError, KeyError, AttributeError)):
# 		algo.get_moves(arr, do_ejection=False)


# Z* (bottleneck lower bound) tests


class TestZstarLowerBound:
    """Verify the LBAP lower bound and Z* computations."""

    @staticmethod
    def _make_array_and_target(shape, target_size, fill_prob=0.8):
        """Build a 3D matrix and target for Z* functions."""
        random.seed(0)
        np.random.seed(0)
        rows, cols = shape
        matrix = (np.random.rand(rows, cols) < fill_prob).astype(int)
        target = np.zeros((rows, cols), dtype=int)
        r0 = (rows - target_size) // 2
        c0 = (cols - target_size) // 2
        target[r0 : r0 + target_size, c0 : c0 + target_size] = 1
        matrix_3d = matrix.reshape(rows, cols, 1)
        target_3d = target.reshape(rows, cols, 1)
        return matrix_3d, target_3d

    def test_zstar_better_returns_nonnegative(self):
        matrix, target = self._make_array_and_target((8, 8), 4)
        zstar = calculate_Zstar_better(matrix, target, n_species=1, metric="euclidean")
        assert zstar >= 0, "Z* must be non-negative"

    def test_zstar_better_grid_metric(self):
        matrix, target = self._make_array_and_target((8, 8), 4)
        zstar_grid = calculate_Zstar_better(matrix, target, n_species=1, metric="moves")
        assert zstar_grid >= 0, "Z* (grid) must be non-negative"

    def test_zstar_better_euclidean_leq_grid(self):
        """Euclidean Z* <= grid Z* because diagonal moves are shorter."""
        matrix, target = self._make_array_and_target((8, 8), 4)
        zstar_euc = calculate_Zstar_better(
            matrix, target, n_species=1, metric="euclidean"
        )
        zstar_grid = calculate_Zstar_better(matrix, target, n_species=1, metric="moves")
        assert (
            zstar_euc <= zstar_grid + 1e-9
        ), f"Euclidean Z* ({zstar_euc}) should be <= grid Z* ({zstar_grid})"

    def test_zstar_original_returns_triple(self):
        matrix, target = self._make_array_and_target((8, 8), 4)
        result = calculate_Zstar(matrix, target[:, :, 0], n_species=1)
        assert len(result) == 3, "calculate_Zstar must return (zstar, LB, flag)"
        zstar, lb, fail_flag = result
        if not fail_flag:
            assert zstar >= lb, f"Z* ({zstar}) must be >= lower bound ({lb})"

    def test_lb_leq_zstar(self):
        """The lower bound from calculate_LB must not exceed calculate_Zstar."""
        matrix, target = self._make_array_and_target((8, 8), 4)
        lb, _ = calculate_LB(matrix, target[:, :, 0], n_species=1)
        zstar, zstar_lb, fail_flag = calculate_Zstar(
            matrix, target[:, :, 0], n_species=1
        )
        if not fail_flag:
            assert lb <= zstar + 1e-9, f"LB ({lb}) must not exceed Z* ({zstar})"

    def test_get_zstar_lower_bound_on_identity(self):
        """For an identity cost matrix, the lower bound is 1."""
        cost = np.eye(5)
        lb = get_Zstar_lower_bound(cost)
        assert lb == 0.0, "Identity cost matrix has zero off-diagonal, LB should be 0"

    def test_zstar_trivial_no_moves(self):
        """When all atoms already sit on target sites, Z* should be zero."""
        target = np.zeros((6, 6, 1), dtype=int)
        target[1:5, 1:5, 0] = 1
        matrix = target.copy()
        zstar = calculate_Zstar_better(matrix, target, n_species=1, metric="euclidean")
        assert zstar == 0.0, "Z* should be 0 when config equals target"

    def test_zstar_scales_with_displacement(self):
        """Shifting atoms further from target increases Z*."""
        np.random.seed(7)
        base_target = np.zeros((10, 10, 1), dtype=int)
        base_target[3:7, 3:7, 0] = 1

        # Close atoms: fill near target
        close = np.zeros((10, 10, 1), dtype=int)
        close[2:8, 2:8, 0] = 1
        zstar_close = calculate_Zstar_better(
            close, base_target, n_species=1, metric="euclidean"
        )

        # Far atoms: fill only edges
        far = np.zeros((10, 10, 1), dtype=int)
        far[0, :, 0] = 1
        far[9, :, 0] = 1
        far[:, 0, 0] = 1
        far[:, 9, 0] = 1
        zstar_far = calculate_Zstar_better(
            far, base_target, n_species=1, metric="euclidean"
        )

        assert zstar_far >= zstar_close, "Displacing atoms further must not decrease Z*"
