# Core utilities for initializing and analyzing atom arrays

import math
import numpy as np
from enum import IntEnum
from numpy.typing import NDArray
from typing import Optional, Tuple

###########
# Classes #
###########


class Configurations(IntEnum):
    """
    Enumeration class for predefined atomic configuration patterns.

    This enum is used in conjunction with `AtomArray.generate_target()`
    and `generate_target_config()` to prepare common patterns of atoms.

    Attributes
    ----------
    ZEBRA_HORIZONTAL : int
        Horizontal zebra stripe pattern configuration.
    ZEBRA_VERTICAL : int
        Vertical zebra stripe pattern configuration.
    CHECKERBOARD : int
        Alternating checkerboard pattern configuration.
    MIDDLE_FILL : int
        Configuration that fills atoms from the middle.
    Left_Sweep : int
        Configuration that sweeps atoms from the left.
    SEPARATE : int
        Separation configuration for dual-species arrangements only.
    RANDOM : int
        Random arrangement configuration.

    Examples
    --------
    >>> config = Configurations.CHECKERBOARD
    >>> atom_array.generate_target(config)
    """

    ZEBRA_HORIZONTAL = 0
    ZEBRA_VERTICAL = 1
    CHECKERBOARD = 2
    MIDDLE_FILL = 3
    Left_Sweep = 4
    SEPARATE = 5  # for dual-species only
    RANDOM = 6


CONFIGURATION_PLOT_LABELS = {
    Configurations.ZEBRA_HORIZONTAL: "Horizontal zebra stripes",
    Configurations.ZEBRA_VERTICAL: "Vertical zebra stripes",
    Configurations.CHECKERBOARD: "Checkerboard",
    Configurations.MIDDLE_FILL: "Middle fill rectangle",
    Configurations.Left_Sweep: "Left Sweep",
    Configurations.RANDOM: "Random",
}


class PhysicalParams:
    """
    Class used to store various physical parameters corresponding to atom, array and optical tweezer properties.
    Parameters
    ----------
    AOD_speed : float, optional
        The speed of the moving tweezers, in um/us. Default is 0.1.
    spacing : float, optional
        Spacing between adjacent atoms in the square array, in m. Default is 5e-6.
    loading_prob : float, optional
        The probability that a single site will be filled during loading.
        Must be in range [0, 1]. Default is 0.6.
    target_occup_prob : float, optional
        If the target configuration is random, the probability that a site in the
        configuration will be occupied by an atom. Must be in range [0, 1].
        Default is 0.5.
    Attributes
    ----------
        Spacing between adjacent atoms in the square array, in m.
        The probability that a single site will be filled during loading.
        The probability that a site in the target configuration will be occupied.
        The speed of the moving tweezers, in um/us.
    Raises
    ------
    ValueError
        If `loading_prob` is not in range [0, 1].
    ValueError
        If `target_occup_prob` is not in range [0, 1].
    """

    def __init__(
        self,
        AOD_speed: float = 0.1,
        spacing: float = 5e-6,
        loading_prob: float = 0.6,
        target_occup_prob: float = 0.5,
    ) -> None:
        # array parameters
        self.spacing = spacing
        if loading_prob > 1 or loading_prob < 0:
            raise ValueError("Variable `loading_prob` must be in range [0,1].")
        if target_occup_prob > 1 or target_occup_prob < 0:
            raise ValueError("Variable `target_occup_prob` must be in range [0,1].")
        self.loading_prob = loading_prob
        self.target_occup_prob = target_occup_prob

        # tweezer parameters
        self.AOD_speed = AOD_speed


class ArrayGeometry(IntEnum):
    """
    Class used to specify the geometry of the atom array.

    This is used in `AtomArray` to determine the geometry of the array.
    See references [LattPy](https://lattpy.readthedocs.io/en/latest/) for examples of other geometries that could be implemented.

    Currently only supports square geometry, but more geometries may be added in the future.

    Attributes
    ----------
    SQUARE : int
        Square lattice geometry.
    RECTANGULAR : int
        Rectangular lattice geometry. NOT SUPPORTED YET (NSY); see CONTRIBUTING.md.
    TRIANGULAR : int
        Triangular lattice geometry. NSY.
    BRAVAIS : int
        Bravais lattice geometry. NSY.
    DECORATED_BRAVAIS : int
        Decorated Bravais lattice geometry. NSY.

    Examples
    --------
    >>> geometry = ArrayGeometry.SQUARE
    >>> atom_array = AtomArray(geometry=geometry)
    """

    SQUARE = 0
    RECTANGULAR = 1  # NOT SUPPORTED YET (NSY); see CONTRIBUTING.md
    TRIANGULAR = 2  # NSY
    BRAVAIS = 3  # NSY
    DECORATED_BRAVAIS = 4  # NSY
    RECTANGLE_TALL = (
        5  # rectangle taller than it is wide, useful for rectangular target configs
    )


class ArrayGeometrySpec:
    """Specification for desired loading geometry.

    - kind: an ArrayGeometry member.
    - params: optional dict of parameters (algorithm-specific).
    """

    kind: ArrayGeometry
    params: Optional[dict] = None

    def __init__(self, kind: ArrayGeometry, params: Optional[dict] = None) -> None:
        self.kind = kind
        self.params = params


#############
# Functions #
#############


def _coerce_rng(rng: np.random.Generator | None) -> np.random.Generator:
    """Return a numpy Generator, deriving deterministic seeds from np.random."""
    if rng is not None:
        return rng

    # Preserve test reproducibility when callers rely on `np.random.seed(...)`.
    seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    return np.random.default_rng(seed)


def random_loading(
    size, probability: float, rng: np.random.Generator | None = None
) -> NDArray:
    """
    Function used to generate initial atom array config.

    Parameters
    ----------
    size : list
        A list of integers specifying the dimensions of the array to be generated. For example, [5,5] would generate a 5x5 array.
    probability : float
        The probability that a given site in the array will be occupied by an atom. Must be in range [0, 1].

    Returns
    -------
    np.ndarray
        A Numpy array of the specified size, where each element is either 0 (unoccupied) or 1 (occupied).
    """
    if len(size) < 2:
        raise ValueError(f"`size` must have at least 2 entries; got {size}.")

    n0 = int(size[0])
    n1 = int(size[1])

    if probability < 0 or probability > 1:
        raise ValueError(f"`probability` must be in [0,1]; got {probability}.")

    rng = _coerce_rng(rng)

    if probability == 0:
        return np.zeros((n0, n1), dtype=np.uint8)
    if probability == 1:
        return np.ones((n0, n1), dtype=np.uint8)
    x = rng.random((n0, n1))
    return (x > (1.0 - float(probability))).astype(np.uint8)


def generate_random_init_target_configs(
    n_shots: int,
    load_prob: float,
    max_sys_size: int,
    target_config=None,
    rng: np.random.Generator | None = None,
) -> Tuple[list, list]:
    """
    Generates random initial and target configurations for atom arrays.

    Parameters
    ----------
    n_shots: int
        The number of initial and target configurations to generate.
    load_prob: float
        The probability that a given site in the initial configuration will be occupied by an atom. Must be in range [0, 1].
    max_sys_size: int
        The size of atom array.
    target_config: Configurations, optional
        Specify the pattern of the target. If set to `Configurations.RANDOM', it generate target based on load_prob.
    """
    rng = _coerce_rng(rng)
    init_config_storage = []
    target_config_storage = []

    for _ in range(n_shots):
        initial_config = random_loading(
            [max_sys_size, max_sys_size], load_prob, rng=rng
        )
        init_config_storage.append(initial_config)

        if target_config == [Configurations.RANDOM]:
            target = random_loading(
                [max_sys_size, max_sys_size], load_prob - 0.1, rng=rng
            )
            target_config_storage.append(target)

    return init_config_storage, target_config_storage


def generate_random_init_configs(
    n_shots: int,
    load_prob: float,
    max_sys_size: int | None = None,
    n_species: int = 1,
    rng: np.random.Generator | None = None,
    shape: list | tuple | None = None,
) -> list:
    """
    Generates random initial atom array configurations.

    Parameters
    ----------
    n_shots : int
        The number of configurations (shots) to generate.
    load_prob : float
        The probability of an individual site being occupied by an atom.
    max_sys_size : int, optional
        The row and column size of a square target array. Ignored if `shape` is provided.
        Either `max_sys_size` or `shape` must be specified.
    n_species : int, optional
        The number of atomic species (1 or 2). Default is 1.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.
    shape : list or tuple, optional
        Rectangular shape as [rows, cols]. If provided, `max_sys_size` is ignored.
        Either `max_sys_size` or `shape` must be specified.

    Returns
    -------
    list of numpy.ndarray
        A list of generated configurations. If `n_species` is 1, arrays are 2D.
        If `n_species` is 2, arrays are 3D with shape (..., ..., 2).

    Raises
    ------
    ValueError
        If `n_species` is not 1 or 2, or if neither `max_sys_size` nor `shape` is provided.
    """
    # Determine array dimensions
    if shape is not None:
        rows, cols = int(shape[0]), int(shape[1])
    elif max_sys_size is not None:
        rows, cols = max_sys_size, max_sys_size
    else:
        raise ValueError(
            "Either `max_sys_size` (for square arrays) or `shape` (for rectangular) must be provided."
        )

    rng = _coerce_rng(rng)

    init_config_storage = []

    for _ in range(n_shots):
        if n_species == 1:
            initial_config = random_loading([rows, cols], load_prob, rng=rng)

        elif n_species == 2:
            initial_config = np.zeros((rows, cols, 2), dtype=np.uint8)

            dual_species_prob = 2 - 2 * math.sqrt(1 - load_prob)
            p_each = dual_species_prob / 2

            initial_config[:, :, 0] = random_loading([rows, cols], p_each, rng=rng)
            initial_config[:, :, 1] = random_loading([rows, cols], p_each, rng=rng)

            # Resolve double-occupancy sites by dropping one species uniformly at random.
            both = (initial_config[:, :, 0] == 1) & (initial_config[:, :, 1] == 1)
            if np.any(both):
                idx = np.argwhere(both)  # shape (k, 2)
                drop = rng.integers(
                    0, 2, size=idx.shape[0]
                )  # which species channel to drop
                initial_config[idx[:, 0], idx[:, 1], drop] = 0

        else:
            raise ValueError(
                f"Argument `n_species` must be either 1 or 2; the provided value is {n_species}."
            )

        init_config_storage.append(initial_config)

    return init_config_storage


def generate_random_target_configs(
    n_shots: int,
    targ_occup_prob: float,
    shape: list,
    rng: np.random.Generator | None = None,
):
    """
    Generates random target configurations with specified occupation probability.

    Creates multiple random target configurations where each site has an independent
    probability of being occupied by an atom.

    Parameters
    ----------
    n_shots : int
        The number of target configurations to generate.
    targ_occup_prob : float
        The probability that a given site in the target configuration will be occupied
        by an atom. Must be in range [0, 1].
    shape : list
        A list of integers specifying the dimensions of each configuration.
        For example, [5, 5] generates 5x5 arrays.

    Returns
    -------
    list of np.ndarray
        A list of generated target configurations. Each element is a 2D numpy array
        where each element is either 0 (unoccupied) or 1 (occupied).

    Examples
    --------
    >>> configs = generate_random_target_configs(10, 0.5, [5, 5])
    >>> len(configs)
    10
    >>> configs[0].shape
    (5, 5)
    """
    rng = _coerce_rng(rng)
    target_config_storage = []
    for _ in range(n_shots):
        target = random_loading(shape, targ_occup_prob, rng=rng)
        target_config_storage.append(target)
    return target_config_storage


def count_atoms_in_columns(matrix: NDArray) -> list:
    """
    Counts the number of atoms in each column of a matrix.

    Iterates through each column and sums the number of occupied sites (value of 1)
    in that column.

    Parameters
    ----------
    matrix : np.ndarray or list of lists
        A 2D array representing the atom configuration where 1 indicates an occupied
        site and 0 indicates an empty site.

    Returns
    -------
    list of int
        A list where the i-th element is the number of atoms in the i-th column.

    Examples
    --------
    >>> matrix = [[1, 0, 1], [0, 1, 0], [1, 1, 0]]
    >>> count_atoms_in_columns(matrix)
    [2, 2, 1]
    """
    return np.sum(np.asarray(matrix), axis=0).tolist()


def left_right_atom_in_row(row: int, direction: int) -> int | None:
    """
    Finds the leftmost or rightmost atom in a row.

    Scans a row in the specified direction to locate the first occupied site.

    Parameters
    ----------
    row : np.ndarray or list
        A 1D array representing a single row where 1 indicates an occupied site
        and 0 indicates an empty site.
    direction : int
        The direction to search: 1 for rightmost atom (forward scan),
        -1 for leftmost atom (reverse scan).

    Returns
    -------
    int or None
        The column index of the first atom found in the specified direction,
        or None if no atom is found in the row.

    Examples
    --------
    >>> row = [0, 1, 0, 1, 0]
    >>> left_right_atom_in_row(row, 1)  # leftmost (rightward scan)
    1
    >>> left_right_atom_in_row(row, -1)  # rightmost (leftward scan)
    3
    """
    row_arr = np.asarray(row)
    occupied = np.flatnonzero(row_arr == 1)
    if occupied.size == 0:
        return None
    # direction convention preserved from old implementation:
    # direction=1 -> rightmost (via [::-1]); direction=-1 -> leftmost
    return int(occupied[-1] if direction == 1 else occupied[0])


def top_bot_atom_in_col(col, direction):
    """
    Finds the topmost or bottommost atom in a column.

    Scans a column in the specified direction to locate the first occupied site.

    Parameters
    ----------
    col : np.ndarray or list
        A 1D array representing a single column where 1 indicates an occupied site
        and 0 indicates an empty site.
    direction : int
        The direction to search: 1 for topmost atom (forward scan),
        -1 for bottommost atom (reverse scan).

    Returns
    -------
    int or None
        The row index of the first atom found in the specified direction,
        or None if no atom is found in the column.

    Examples
    --------
    >>> col = [0, 1, 0, 1, 0]
    >>> top_bot_atom_in_col(col, 1)  # topmost (downward scan)
    1
    >>> top_bot_atom_in_col(col, -1)  # bottommost (upward scan)
    3
    """
    col_arr = np.asarray(col)
    occupied = np.flatnonzero(col_arr == 1)
    if occupied.size == 0:
        return None
    return int(occupied[-1] if direction == 1 else occupied[0])


def find_lowest_atom_in_col(col: int) -> int | None:
    """
    Finds the lowest (bottom-most) atom in a column.

    Scans a column from bottom to top and returns the index of the lowest occupied site.

    Parameters
    ----------
    col : np.ndarray or list
        A 1D array representing a single column where 1 indicates an occupied site
        and 0 indicates an empty site. Index 0 is the top, increasing downward.

    Returns
    -------
    int or None
        The row index of the lowest atom in the column, or None if the column
        contains no atoms.

    Examples
    --------
    >>> col = [0, 1, 0, 1, 0]
    >>> find_lowest_atom_in_col(col)
    3
    """
    col_arr = np.asarray(col)
    occupied = np.flatnonzero(col_arr == 1)
    if occupied.size == 0:
        return None
    return int(occupied[-1])


def get_move_distance(
    from_row: int, from_col: int, to_row: int, to_col: int, spacing: float = 5e-6
) -> float:
    """
    Calculates the physical distance for moving an atom from one position to another.

    Computes the Manhattan distance (L1 norm) between two positions in the array
    and converts it to physical units using the lattice spacing.

    Parameters
    ----------
    from_row : int
        The row index of the starting position.
    from_col : int
        The column index of the starting position.
    to_row : int
        The row index of the ending position.
    to_col : int
        The column index of the ending position.
    spacing : float, optional
        The physical distance between adjacent lattice sites, in meters.
        Default is 5e-6 m (5 micrometers).

    Returns
    -------
    float
        The physical distance between the two positions, in meters.

    Examples
    --------
    >>> get_move_distance(0, 0, 2, 3, spacing=5e-6)
    2.5e-05
    >>> get_move_distance(1, 1, 1, 4, spacing=1e-6)
    3e-06
    """
    move_distance = (abs(from_row - to_row) + abs(from_col - to_col)) * spacing
    return move_distance


def atom_loss(
    matrix: np.ndarray,
    move_time: float,
    lifetime: float = 30,
    rng: np.random.Generator | None = None,
) -> Tuple[NDArray, bool]:
    """
    Sample atom loss over a finite evolution time.

    Parameters
    ----------
    matrix : np.ndarray
        Occupancy array.
    move_time : float
        Evolution time.
    lifetime : float, optional
        Vacuum-limited lifetime of a single atom in a tweezer.
    rng : np.random.Generator | None, optional
        Random number generator.


    Returns
    -------
    np.ndarray
        Post-loss occupancy array.
    bool
        ``True`` if at least one atom was lost.

    Raises
    ------
    ValueError
        If ``lifetime`` is nonpositive or ``matrix`` has unsupported rank.
    """
    if lifetime <= 0:
        raise ValueError(f"`lifetime` must be > 0; got {lifetime}.")

    p_survive = float(np.exp(-move_time / lifetime))
    rng = _coerce_rng(rng)

    # Build a 2D survival mask
    mask2d = random_loading(list(np.shape(matrix)), p_survive, rng=rng)

    if matrix.ndim == 2:
        mask = mask2d
    elif matrix.ndim == 3:
        mask = mask2d[:, :, None]  # broadcast over species axis
    else:
        raise ValueError(f"`matrix` must be 2D or 3D; got shape {matrix.shape}.")

    matrix_copy = np.asarray(matrix).copy()
    matrix_copy = matrix_copy * mask

    loss_flag = bool(np.any(matrix_copy != matrix))
    return matrix_copy, loss_flag


def atom_loss_dual(
    matrix: NDArray,
    move_time: float,
    lifetime: float = 30,
    rng: np.random.Generator | None = None,
) -> Tuple[NDArray, bool]:
    """
    Sample atom loss for a dual-species array.

    Parameters
    ----------
    matrix : np.ndarray
        Dual-species occupancy array of shape ``(rows, cols, 2)``.
    move_time : float
        Evolution time.
    lifetime : float, optional
        Vacuum-limited lifetime of a single atom in a tweezer.
    rng : np.random.Generator | None, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Post-loss occupancy array.
    bool
        ``True`` if at least one atom was lost.

    Raises
    ------
    ValueError
        If ``matrix`` is not a dual-species array or ``lifetime`` is nonpositive.
    """
    if lifetime <= 0:
        raise ValueError(f"`lifetime` must be > 0; got {lifetime}.")

    if np.asarray(matrix).ndim != 3 or np.asarray(matrix).shape[-1] != 2:
        raise ValueError(
            f"`matrix` must have shape (rows, cols, 2); got {np.shape(matrix)}."
        )

    return atom_loss(matrix, move_time, lifetime=lifetime, rng=rng)


def count_atoms_in_row(row: int) -> int:
    """
    Counts the total number of atoms in a row.

    Sums the values in the row array, counting each occupied site (value of 1).

    Parameters
    ----------
    row : np.ndarray or list
        A 1D array representing a single row where 1 indicates an occupied site
        and 0 indicates an empty site.

    Returns
    -------
    int or float
        The total number of atoms (occupied sites) in the row.

    Examples
    --------
    >>> row = [1, 0, 1, 1, 0]
    >>> count_atoms_in_row(row)
    3
    """
    return np.sum(row)


def calculate_filling_fraction(atom_count: int, row_length: int) -> float:
    """
    Calculates the filling fraction of a row as a percentage.

    Computes what percentage of sites in a row are occupied by atoms.

    Parameters
    ----------
    atom_count : int or float
        The number of atoms in the row.
    row_length : int
        The total number of sites in the row.

    Returns
    -------
    float
        The filling fraction as a percentage (0 to 100).

    Examples
    --------
    >>> calculate_filling_fraction(3, 5)
    60.0
    >>> calculate_filling_fraction(2, 10)
    20.0
    """
    return (atom_count / row_length) * 100


def save_frames(temp_frames: list, combined_frames: list) -> Tuple[list, list]:
    """
    Saves temporary frames to the combined frames list and clears the temporary storage.

    Extends the combined frames list with frames from the temporary buffer and
    clears the temporary buffer for reuse.

    Parameters
    ----------
    temp_frames : list
        A temporary list of animation frames to be saved.
    combined_frames : list
        The main list that accumulates all frames.

    Returns
    -------
    tuple of (list, list)
        A tuple containing (cleared_temp_frames, updated_combined_frames).

    Examples
    --------
    >>> temp = [1, 2, 3]
    >>> combined = []
    >>> temp_new, combined_new = save_frames(temp, combined)
    >>> combined_new
    [1, 2, 3]
    >>> temp_new
    []
    """
    combined_frames.extend(temp_frames)
    temp_frames.clear()
    return temp_frames, combined_frames


def generate_middle_fifty(length: int, filling_threshold: float = 0.5) -> list[int]:
    """
    Generates a smaller square array dimension that fills the middle of a larger array.

    Calculates the maximum size of a centered square that occupies at most a specified
    fraction of the total array area.

    Parameters
    ----------
    length : int
        The size of the original square array (length x length).
    filling_threshold : float, optional
        The maximum fraction of the array to be filled, in range [0, 1].
        Default is 0.5.

    Returns
    -------
    list of int
        A list [max_L, max_L] representing the size of the middle fill rectangle.

    Notes
    -----
    Currently only works for square arrays. Generalization to rectangular arrays
    is needed (see TODO in source code).

    Examples
    --------
    >>> generate_middle_fifty(10, 0.5)
    [7, 7]
    >>> generate_middle_fifty(20, 0.25)
    [14, 14]
    """
    # TODO this only works for square arrays, generalize to rectangular
    max_L = length
    while (max_L**2) / (length**2) >= filling_threshold:
        max_L -= 1
    return [max_L, max_L]


def array_shape_for_geometry(
    geometry_spec, target_size: int, loading_prob: float = 0.6
) -> tuple[int, int]:
    """Return (rows, cols) for a loading array given `geometry_spec`.

    Accepted `geometry_spec` types:
    - None: square fallback (see rules).
    - `ArrayGeometrySpec` instance: follows `kind`.
    - tuple/list of two ints: interpreted as (rows, cols).
    """
    try:
        t = int(target_size)
        if t <= 0:
            raise ValueError
    except Exception:
        raise ValueError("target_size must be a positive integer.") from None

    # If a plain two-int tuple/list provided -> coerce
    if isinstance(geometry_spec, (list, tuple)) and len(geometry_spec) == 2:
        try:
            rows = int(geometry_spec[0])
            cols = int(geometry_spec[1])
        except Exception:
            raise ValueError("Geometry tuple/list must contain two integers.") from None
        rows = max(rows, t)
        cols = max(cols, t)
        return rows, cols

    # If None or SQUARE: square fallback scaled by loading_prob
    if geometry_spec is None or (
        isinstance(geometry_spec, ArrayGeometrySpec)
        and geometry_spec.kind == ArrayGeometry.SQUARE
    ):
        side = int(math.ceil(math.sqrt(t)))
        scale = (
            int(math.ceil(1.0 / math.sqrt(float(loading_prob))))
            if loading_prob and loading_prob > 0
            else 1
        )
        side = side * scale
        # Ensure some donor margin around the target so rearrangement algorithms
        # can source atoms from outside the target region. Make array at least
        # target_size + 2 on a side.
        side = max(side, t + 2)
        return side, side

    # ArrayGeometrySpec handling
    if isinstance(geometry_spec, ArrayGeometrySpec):
        if geometry_spec.kind == ArrayGeometry.RECTANGLE_TALL:
            params = geometry_spec.params or {}
            preferred_width_factor = float(params.get("preferred_width_factor", 2.0))
            min_extra_columns = int(params.get("min_extra_columns", 2))
            rows = int(t)
            cols = int(math.ceil(t * preferred_width_factor))
            cols = max(cols, t + min_extra_columns)
            rows = max(rows, t)
            cols = max(cols, t)
            return rows, cols
        # Fallback: unknown kinds -> square fallback
        side = int(math.ceil(math.sqrt(t)))
        scale = (
            int(math.ceil(1.0 / math.sqrt(float(loading_prob))))
            if loading_prob and loading_prob > 0
            else 1
        )
        side = side * scale + 2
        side = max(side, t)
        return side, side

    raise ValueError(
        "Unsupported `geometry_spec` type; expected ArrayGeometrySpec, (rows,cols), or None."
    )
