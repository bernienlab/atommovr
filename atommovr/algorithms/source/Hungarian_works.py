import numpy as np
import copy
from collections import deque
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

from atommovr.algorithms.Algorithm_class import Algorithm
from atommovr.utils.core import random_loading, generate_middle_fifty, Configurations
from atommovr.utils.move_utils import (
    Move,
    move_atoms,
    move_atoms_noiseless,
    get_move_list_from_AOD_cmds,
    get_AOD_cmds_from_move_list,
)
from atommovr.algorithms.source.ejection import ejection
from atommovr.algorithms.source.scaling_lower_bound import make_cost_matrix_square
from atommovr.algorithms.source.PPSU_weight_matching import bttl_threshold


def parallel_LBAP_algorithm_works(
    atom_arrays: np.ndarray,
    target_config: np.ndarray,
    do_ejection: bool = False,
    round_lim: int = 15,
):
    # Initialize the variables
    LBAP_success_flag = False
    complete_flag = False
    move_set = []
    matrix = copy.deepcopy(atom_arrays)
    round_count = 0

    while (not complete_flag) and (round_count < round_lim):
        # print(f"Got here_{round_count}")
        N_independent_moves_path = []
        # 1. Generate the assignments
        prepared_assignments = generate_LBAP_assignments(matrix, target_config)

        # 2. Find out N independent paths
        for start, target in prepared_assignments:
            single_move_path = generate_path(matrix, start, target)
            if single_move_path == []:
                pass
            # Decompose the single_move_path into independent moves of several obstacle atoms
            else:
                N_independent_moves_path.append(single_move_path)

        # 3. Transform the N_independent_moves_path into a list of moves
        matrix, Hung_parallel_move_set = transform_paths_into_moves(
            matrix, N_independent_moves_path
        )
        move_set.extend(Hung_parallel_move_set)

        # effective_config = np.multiply(matrix, target_config)
        if Algorithm.get_success_flag(
            matrix, target_config, do_ejection=do_ejection, n_species=1
        ):
            complete_flag = True
            LBAP_success_flag = True
        round_count += 1

    # 4. Eject to certain geoemetry
    if do_ejection:
        eject_moves, eject_config = ejection(
            matrix, target_config, [0, len(matrix) - 1, 0, len(matrix[0]) - 1]
        )
        move_set.extend(eject_moves)
    else:
        eject_config = matrix

    return eject_config, move_set, LBAP_success_flag


def generate_LBAP_assignments(matrix, target_config):

    # Define target positions for the center square in a matrix.
    current_positions, target_positions = define_current_and_target(
        matrix, target_config
    )

    # Generate the cost matrix using the current atom positions and the target positions
    cost_matrix = generate_cost_matrix(current_positions, target_positions)

    sq_cost = make_cost_matrix_square(cost_matrix)

    max_val = np.max(sq_cost)
    reverse_cost_mat = np.zeros_like(sq_cost)
    for i in range(len(reverse_cost_mat)):
        for j in range(len(reverse_cost_mat[0])):
            reverse_cost_mat[i, j] = max_val + 1 - sq_cost[i, j]

    sparsemat = csr_matrix(reverse_cost_mat)
    result_dict = bttl_threshold(
        sparsemat.indptr,
        sparsemat.indices,
        sparsemat.data,
        sparsemat.shape[0],
        sparsemat.shape[1],
    )
    col_inds = result_dict["match"]
    col_ind = []
    row_ind = []
    for c_ind in range(len(col_inds)):
        col = col_inds[c_ind]
        row = c_ind
        try:
            cost_matrix[row, col]
            col_ind.append(col)
            row_ind.append(row)
        except IndexError:
            pass
    # costs = []
    # for row_ind in range(len(sq_cost)):
    #     col_ind = col_inds[row_ind]
    #     costs.append(sq_cost[row_ind, col_ind])

    # Pair up row_ind and col_ind and sort by col_ind
    paired_indices = sorted(zip(row_ind, col_ind, strict=True), key=lambda x: x[1])

    if paired_indices:
        # Unzip the sorted pairs if paired_indices is not empty
        sorted_row_ind, sorted_col_ind = zip(*paired_indices, strict=True)
    else:
        # Assign default values if paired_indices is empty
        sorted_row_ind, sorted_col_ind = [], []

    prepared_assignments = [
        (current_positions[i], target_positions[j])
        for i, j in zip(sorted_row_ind, sorted_col_ind, strict=True)
    ]

    return prepared_assignments


def Hungarian_algorithm_works_fast(
    atom_arrays: np.ndarray,
    target_config: np.ndarray,
    do_ejection: bool = False,
    final_size: list[int] | None = None,
):
    """
    Execute the Hungarian rearrangement routine with lower Python overhead.

    Parameters
    ----------
    atom_arrays : np.ndarray
        Current occupancy grid.
    target_config : np.ndarray
        Desired occupancy grid.
    do_ejection : bool, optional
        Whether to append the optional ejection stage.
    final_size : list[int] | None, optional
        Optional geometry bounds for ejection.

    Returns
    -------
    tuple[np.ndarray, list, bool]
        Final configuration, move list, and success flag.
    """
    from atommovr.algorithms.Algorithm_class import Algorithm
    from atommovr.algorithms.source.ejection import ejection

    move_set: list = []
    matrix: np.ndarray = atom_arrays.copy()

    if final_size is None:
        final_size = [0, len(matrix[0]) - 1, 0, len(matrix) - 1]

    prepared_assignments = generate_assignments_fast(matrix, target_config, final_size)

    for start, target in prepared_assignments:
        hungarian_move = move_atom_and_show_grid(matrix, start, target)
        move_set.extend(hungarian_move)

    if do_ejection:
        eject_moves, eject_config = ejection(matrix, target_config, final_size)
        move_set.extend(eject_moves)
    else:
        eject_config = matrix.copy()

    success_flag: bool = Algorithm.get_success_flag(
        eject_config.reshape(np.shape(target_config)),
        target_config,
        do_ejection=do_ejection,
        n_species=1,
    )
    return eject_config, move_set, success_flag


def Hungarian_algorithm_works(
    atom_arrays: np.ndarray,
    target_config: np.ndarray,
    do_ejection: bool = False,
    final_size: list[int] | None = None,
):
    move_set = []
    matrix = copy.deepcopy(atom_arrays)
    if final_size is None:
        final_size = [0, len(matrix[0]) - 1, 0, len(matrix) - 1]
    elif len(final_size) == 0:
        final_size = [0, len(matrix[0]) - 1, 0, len(matrix) - 1]

    # Define target positions for the center square in a matrix.
    current_positions, target_positions = define_current_and_target(
        matrix, target_config
    )

    # Generate the cost matrix using the current atom positions and the target positions
    cost_matrix = generate_cost_matrix(current_positions, target_positions)

    # row_ind and col_ind are arrays of indices indicating the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Pair up row_ind and col_ind and sort by col_ind
    paired_indices = sorted(zip(row_ind, col_ind, strict=True), key=lambda x: x[1])

    if paired_indices:
        # Unzip the sorted pairs if paired_indices is not empty
        sorted_row_ind, sorted_col_ind = zip(*paired_indices, strict=True)
    else:
        # Assign default values if paired_indices is empty
        sorted_row_ind, sorted_col_ind = [], []
    prepared_assignments = [
        (current_positions[i], target_positions[j])
        for i, j in zip(sorted_row_ind, sorted_col_ind, strict=True)
    ]

    for start, target in prepared_assignments:
        Hungarian_move = []
        Hungarian_move = move_atom_and_show_grid_og(matrix, start, target)
        move_set.extend(Hungarian_move)

    # Optional ejection argument
    if do_ejection:
        eject_moves, eject_config = ejection(matrix, target_config, final_size)
        move_set.extend(eject_moves)
    else:
        eject_config = copy.deepcopy(matrix)

    success_flag = Algorithm.get_success_flag(
        eject_config.reshape(np.shape(target_config)),
        target_config,
        do_ejection=do_ejection,
        n_species=1,
    )

    return eject_config, move_set, success_flag


def parallel_Hungarian_algorithm_works(
    atom_arrays: np.ndarray,
    target_config: np.ndarray,
    do_ejection: bool = False,
    final_size: list[int] | None = None,
    round_lim: int = 15,
):
    # Initialize the variables
    Hungarian_success_flag = False
    complete_flag = False
    move_set = []
    matrix = atom_arrays.copy()
    round_count = 0
    if final_size is None:
        final_size = []

    while (not complete_flag) and (round_count < round_lim):
        N_independent_moves_path = []
        # 1. Generate the assignments
        prepared_assignments = generate_assignments_fast(
            matrix, target_config, final_size
        )

        # 2. Find out N independent paths
        for start, target in prepared_assignments:
            single_move_path = generate_path(matrix, start, target)
            if single_move_path == []:
                pass
            # Decompose the single_move_path into independent moves of several obstacle atoms
            else:
                N_independent_moves_path.append(single_move_path)

        # 3. Transform the N_independent_moves_path into a list of moves
        matrix, Hung_parallel_move_set = transform_paths_into_moves_fast(
            matrix, N_independent_moves_path
        )
        move_set.extend(Hung_parallel_move_set)

        # effective_config = np.multiply(matrix, target_config)
        if Algorithm.get_success_flag(
            matrix, target_config, do_ejection=do_ejection, n_species=1
        ):
            complete_flag = True
            Hungarian_success_flag = True
        round_count += 1

    # 4. Eject to certain geoemetry
    if do_ejection:
        eject_moves, eject_config = ejection(
            matrix, target_config, [0, len(matrix) - 1, 0, len(matrix[0]) - 1]
        )
        move_set.extend(eject_moves)
    else:
        eject_config = matrix

    return eject_config, move_set, Hungarian_success_flag


## refactored code for speed


def define_current_and_target_fast(
    matrix: np.ndarray,
    target_config: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Identify movable atom coordinates and unfilled target coordinates.

    Parameters
    ----------
    matrix : np.ndarray
        Current occupancy grid.
    target_config : np.ndarray
        Desired occupancy grid.

    Returns
    -------
    tuple[list[tuple[int, int]], list[tuple[int, int]]]
        Source coordinates and target coordinates as lists of ``(row, col)``
        tuples.
    """
    matrix_2d: np.ndarray = _as_2d_occupancy(matrix, "matrix")
    target_2d: np.ndarray = _as_2d_occupancy(target_config, "target_config")

    if matrix_2d.shape != target_2d.shape:
        raise ValueError(
            f"matrix and target_config must have the same 2D shape; got "
            f"{matrix_2d.shape} and {target_2d.shape}."
        )

    current_positions_arr: np.ndarray = np.argwhere((matrix_2d == 1) & (target_2d == 0))
    target_positions_arr: np.ndarray = np.argwhere((target_2d == 1) & (matrix_2d == 0))

    current_positions: list[tuple[int, int]] = [
        (int(pos[0]), int(pos[1])) for pos in current_positions_arr
    ]
    target_positions: list[tuple[int, int]] = [
        (int(pos[0]), int(pos[1])) for pos in target_positions_arr
    ]

    return current_positions, target_positions


def generate_cost_matrix_fast(
    current_positions: list[tuple[int, int]],
    target_positions: list[tuple[int, int]],
) -> np.ndarray:
    """
    Build the pairwise Euclidean cost matrix for Hungarian assignment.

    Parameters
    ----------
    current_positions : list[tuple[int, int]]
        Coordinates of movable atoms.
    target_positions : list[tuple[int, int]]
        Coordinates of currently empty target sites.

    Returns
    -------
    np.ndarray
        Cost matrix of shape ``(n_atoms, n_targets)``.
    """
    num_atoms: int = len(current_positions)
    num_targets: int = len(target_positions)

    if num_atoms == 0 or num_targets == 0:
        return np.zeros((num_atoms, num_targets), dtype=np.float64)

    current_arr: np.ndarray = np.asarray(current_positions, dtype=np.int64)
    target_arr: np.ndarray = np.asarray(target_positions, dtype=np.int64)

    deltas: np.ndarray = current_arr[:, None, :] - target_arr[None, :, :]
    sq_dist: np.ndarray = np.sum(deltas * deltas, axis=2, dtype=np.int64)

    return np.sqrt(sq_dist, dtype=np.float64)


def generate_assignments_fast(
    matrix: np.ndarray,
    target_config: np.ndarray,
    final_size: list[int],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Generate Hungarian assignments in the same output format as the original.
    """
    if len(final_size) == 0:
        final_size = [0, len(matrix[0]) - 1, 0, len(matrix) - 1]

    current_positions, target_positions = define_current_and_target_fast(
        matrix,
        target_config,
    )
    cost_matrix = generate_cost_matrix_fast(current_positions, target_positions)

    if cost_matrix.size == 0:
        return []

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    order = np.argsort(col_ind)

    prepared_assignments = [
        (current_positions[int(i)], target_positions[int(j)])
        for i, j in zip(row_ind[order], col_ind[order], strict=True)
    ]

    for start, target in prepared_assignments:
        if len(start) != 2 or len(target) != 2:
            raise RuntimeError(
                f"Fast Hungarian produced non-2D coordinate: "
                f"start={start}, target={target}"
            )

    return prepared_assignments


## original code


def generate_assignments(matrix, target_config, final_size):

    if len(final_size) == 0:
        final_size = [0, len(matrix[0]) - 1, 0, len(matrix) - 1]

    # Define target positions for the center square in a matrix.
    current_positions, target_positions = define_current_and_target(
        matrix, target_config
    )

    # Generate the cost matrix using the current atom positions and the target positions
    cost_matrix = generate_cost_matrix(current_positions, target_positions)

    # row_ind and col_ind are arrays of indices indicating the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Pair up row_ind and col_ind and sort by col_ind
    paired_indices = sorted(zip(row_ind, col_ind, strict=True), key=lambda x: x[1])

    if paired_indices:
        # Unzip the sorted pairs if paired_indices is not empty
        sorted_row_ind, sorted_col_ind = zip(*paired_indices, strict=True)
    else:
        # Assign default values if paired_indices is empty
        sorted_row_ind, sorted_col_ind = [], []

    prepared_assignments = [
        (current_positions[i], target_positions[j])
        for i, j in zip(sorted_row_ind, sorted_col_ind, strict=True)
    ]

    return prepared_assignments


def generate_path(arrays, start, end):
    grid = copy.deepcopy(arrays)
    # Initialize current position
    current_pos = start
    path = []

    while current_pos != end:
        path, current_pos = bfs_move_atom_og(grid, current_pos, end, path)

    path = flatten_tuple(path)[::-1]
    grid, path = generate_decomposed_move_set(grid, path)

    return path


def define_current_and_target(matrix, target_config):
    current_positions = [
        (row, col)
        for row in range(len(matrix))
        for col in range(len(matrix[0]))
        if matrix[row][col] == 1
        if target_config[row][col] == 0
    ]  # NKH this should in theory not change anything...
    target_positions = [
        (row, col)
        for row in range(len(matrix))
        for col in range(len(matrix[0]))
        if target_config[row][col] == 1
        if matrix[row][col] == 0
    ]  # same here
    return current_positions, target_positions


# Generate a cost matrix for the Hungarian Algorithm.
def generate_cost_matrix(current_positions, target_positions):
    num_atoms = len(current_positions)
    num_targets = len(target_positions)
    cost_matrix = np.zeros((num_atoms, num_targets))

    for i, current in enumerate(current_positions):
        for j, target in enumerate(target_positions):
            cost_matrix[i, j] = np.sqrt(
                (current[0] - target[0]) ** 2 + (current[1] - target[1]) ** 2
            )
    return cost_matrix


##Move the atom from start to end according to Hungarian assignment
def move_atom_and_show_grid(grid, start, end):
    # Initialize current position
    current_pos = start
    path = []

    while current_pos != end:
        path, current_pos = bfs_move_atom(grid, current_pos, end, path)

    path = flatten_tuple(path)[::-1]
    grid, path = generate_decomposed_move_set(grid, path)

    return path


def move_atom_and_show_grid_og(grid, start, end):
    # Initialize current position
    current_pos = start
    path = []

    while current_pos != end:
        path, current_pos = bfs_move_atom_og(grid, current_pos, end, path)

    path = flatten_tuple(path)[::-1]
    grid, path = generate_decomposed_move_set(grid, path)

    return path


def generate_AOD_cmds(matrix, move_seq):
    # Keep Hungarian-local API while delegating command inference to the
    # canonical converter, then preserve the legacy feasibility checks.
    horiz_AOD_cmds, vert_AOD_cmds, parallel_success_flag = get_AOD_cmds_from_move_list(
        matrix, move_seq, verify=False
    )

    if not parallel_success_flag:
        return horiz_AOD_cmds, vert_AOD_cmds, False

    for move in move_seq:
        if matrix[move.from_row][move.from_col] == 0:
            return horiz_AOD_cmds, vert_AOD_cmds, False

    move_list = get_move_list_from_AOD_cmds(horiz_AOD_cmds, vert_AOD_cmds)
    matrix_from_AOD = move_atoms_noiseless(copy.deepcopy(matrix), move_list)
    matrix_from_seq = move_atoms_noiseless(
        copy.deepcopy(matrix), copy.deepcopy(move_seq)
    )

    if not np.array_equal(matrix_from_AOD, matrix_from_seq):
        parallel_success_flag = False

    return horiz_AOD_cmds, vert_AOD_cmds, parallel_success_flag


def generate_decomposed_move_set(grid, path):
    decomposed_move_set = []

    # Iterate all path segments (((a1,b1), (a2,b2), (a3, b3), (a4,b4)), ((c1,d1), (c2,d2)))
    try:
        for segmant in path:
            from_row, from_col = segmant[0]
            for coordinate in segmant:
                to_row, to_col = coordinate
                # To exclude the frist move
                if from_row != to_row or from_col != to_col:
                    decomposed_move_set.append(
                        [Move(from_row, from_col, to_row, to_col)]
                    )
                    grid[from_row][from_col] = 0
                    grid[to_row][to_col] = 1
                    from_row, from_col = to_row, to_col
            # decomposed_move_set.append(segmant_moves)
    except IndexError:
        return grid, []

    return grid, decomposed_move_set


def regroup_parallel_moves_fast(
    matrix: np.ndarray,
    move_seqq: list[Move],
) -> list[list[Move]]:
    """
    Greedily regroup sequential moves into parallel-executable batches.

    Why this exists
    ---------------
    The legacy implementation repeatedly copies the working matrix, rebuilds
    candidate move lists, and re-computes the same initial atom-count invariant
    inside nested loops. This refactor preserves the same greedy grouping order
    and validation semantics, while moving a few guaranteed-fail checks ahead of
    the expensive AOD/simulation path.

    Parameters
    ----------
    matrix : np.ndarray
        Current occupancy grid.
    move_seqq : list[Move]
        Sequential move list to regroup.

    Returns
    -------
    list[list[Move]]
        Parallel move batches in the same greedy order as the legacy routine.
    """
    matrix_copy: np.ndarray = matrix.copy()
    parallel_seq: list[list[Move]] = []
    parallel_ind_set: set[int] = set()

    current_atom_count: int = int(np.sum(matrix_copy))

    for move_ind, move in enumerate(move_seqq):
        if move_ind in parallel_ind_set:
            continue
        if matrix_copy[move.from_row, move.from_col] == 0:
            continue

        parallel_moves: list[Move] = [move]
        parallel_ind_set.add(move_ind)

        for p_move_ind, p_move in enumerate(move_seqq):
            if p_move_ind in parallel_ind_set:
                continue

            # Safe early reject: the legacy code rejects this candidate anyway,
            # but only after building AOD commands.
            if matrix_copy[p_move.from_row, p_move.from_col] == 0:
                continue

            parallel_moves.append(p_move)
            _, _, can_parallelize = get_AOD_cmds_from_move_list(
                matrix_copy, parallel_moves, verify=True
            )

            if not can_parallelize:
                parallel_moves.pop()
                continue

            scratch: np.ndarray = matrix_copy.copy()
            scratch = move_atoms_noiseless(scratch, parallel_moves)

            if int(np.sum(scratch)) == current_atom_count:
                parallel_ind_set.add(p_move_ind)
            else:
                parallel_moves.pop()

        scratch = matrix_copy.copy()
        scratch = move_atoms_noiseless(scratch, parallel_moves)

        if int(np.sum(scratch)) == current_atom_count:
            parallel_seq.append(parallel_moves.copy())
            matrix_copy = scratch

    return parallel_seq


def regroup_parallel_moves(matrix, move_seqq):
    matrix_copy = copy.deepcopy(matrix)
    parallel_seq = []
    parallel_ind_set = set()

    # Iterate through all size of subset
    for move_ind, move in enumerate(move_seqq):
        if (
            move_ind in parallel_ind_set
            or matrix_copy[move.from_row][move.from_col] == 0
        ):
            continue
        parallel_moves = [move]
        parallel_ind_set.add(move_ind)

        for p_move_ind, p_move in enumerate(move_seqq):

            if p_move_ind in parallel_ind_set:
                continue

            horiz_AOD_cmds, vert_AOD_cmds, can_parallelize = generate_AOD_cmds(
                matrix_copy, parallel_moves + [p_move]
            )

            if not can_parallelize:
                continue
            else:
                parallel_moves_test = parallel_moves + [p_move]
                if matrix_copy[p_move.from_row][p_move.from_col] == 0:
                    can_parallelize = False
                    continue
                sanit_check_matrix = copy.deepcopy(matrix_copy)
                total_atom_num_init = np.sum(sanit_check_matrix)
                matrix_copy = move_atoms_noiseless(matrix_copy, parallel_moves_test)
                total_atom_num_final = np.sum(matrix_copy)

                if total_atom_num_init == total_atom_num_final:
                    parallel_moves += [p_move]
                    parallel_ind_set.add(p_move_ind)
                    matrix_copy = copy.deepcopy(sanit_check_matrix)
                else:
                    matrix_copy = copy.deepcopy(sanit_check_matrix)
                    continue

        sanit_check_matrix = copy.deepcopy(matrix_copy)
        total_atom_num_init = np.sum(sanit_check_matrix)
        matrix_copy = move_atoms_noiseless(matrix_copy, parallel_moves)
        total_atom_num_final = np.sum(matrix_copy)

        if total_atom_num_init == total_atom_num_final:
            parallel_seq.append(parallel_moves)
        else:
            matrix_copy = copy.deepcopy(sanit_check_matrix)

    return parallel_seq


def transform_paths_into_moves_fast(
    matrix: np.ndarray,
    N_independent_moves_path: list[list[list[Move]]],
) -> tuple[np.ndarray, list[list[Move]]]:
    """
    Convert independent path decompositions into executable parallel move rounds.

    This refactor preserves the original scheduling behavior exactly, including
    the original mutation-during-iteration semantics on ``path_in_moves``.
    The optimization here is intentionally conservative: it removes unused
    temporary containers while leaving the control flow unchanged.

    Parameters
    ----------
    matrix : np.ndarray
        Current occupancy grid.
    N_independent_moves_path : list[list[list[Move]]]
        Nested path representation used by the legacy Hungarian code.

    Returns
    -------
    tuple[np.ndarray, list[list[Move]]]
        Updated occupancy grid and grouped parallel move rounds.
    """
    parallel_move_set: list[list[Move]] = []

    # Build only the intersection information that is actually used.
    intersection_set: dict[tuple[int, int], int] = {}

    n_paths: int = len(N_independent_moves_path)
    for i in range(n_paths):
        for j in range(i, n_paths):
            if i != j:
                _, intersections = check_intersection(
                    N_independent_moves_path[i],
                    N_independent_moves_path[j],
                )
                if len(intersections) > 0:
                    for intersection in intersections:
                        if intersection not in intersection_set:
                            intersection_set[intersection] = 0
                        else:
                            intersection_set[intersection] = (
                                intersection_set[intersection] + 1
                            )

    keep_running_flag: bool = True
    count: int = 0

    # Preserve the original hard stop and loop structure.
    while keep_running_flag and count < 5:
        keep_running_flag = True
        moves_in_scan: list[Move] = []
        destination_set: set[tuple[int, int]] = set()

        # IMPORTANT: preserve original semantics exactly.
        # We intentionally iterate over the live list while mutating it via
        # pop(0), because the legacy behavior depends on that.
        for path_in_moves in N_independent_moves_path:
            if len(path_in_moves) > 0:
                for move in path_in_moves:
                    crossing_path_flag: bool = check_crossing_path(
                        matrix,
                        move[0],
                        intersection_set,
                        destination_set,
                        path_in_moves,
                    )
                    if not crossing_path_flag:
                        moves_in_scan.append(move[0])
                        path_in_moves.pop(0)
                        destination_set.add((move[0].to_row, move[0].to_col))
                    else:
                        break

        if len(moves_in_scan) > 0:
            grouped_moves: list[list[Move]] = regroup_parallel_moves_fast(
                matrix, moves_in_scan
            )
            parallel_move_set.extend(grouped_moves)

            for moves in grouped_moves:
                matrix = move_atoms_noiseless(matrix, moves)
                for move in moves:
                    from_coord: tuple[int, int] = (move.from_row, move.from_col)
                    to_coord: tuple[int, int] = (move.to_row, move.to_col)

                    if from_coord in intersection_set:
                        if intersection_set[from_coord] > 0:
                            intersection_set[from_coord] -= 1
                        else:
                            del intersection_set[from_coord]

                    if to_coord in intersection_set:
                        if intersection_set[to_coord] > 0:
                            intersection_set[to_coord] -= 1
                        else:
                            del intersection_set[to_coord]
        else:
            keep_running_flag = False

        count += 1

    return matrix, parallel_move_set


def transform_paths_into_moves(matrix, N_independent_moves_path):
    parallel_move_set = []

    # 1. Build up intersection information for these N independent paths
    intersection_matrix = np.zeros(
        (len(N_independent_moves_path), len(N_independent_moves_path), 1)
    )
    intersection_coordinates = [
        [[] for i in range(len(N_independent_moves_path))]
        for j in range(len(N_independent_moves_path))
    ]
    intersection_set = {}

    for i in range(len(N_independent_moves_path)):
        for j in range(i, len(N_independent_moves_path)):
            if i != j:
                intersection_matrix[i][j], intersection_coordinates[i][j] = (
                    check_intersection(
                        N_independent_moves_path[i], N_independent_moves_path[j]
                    )
                )
                if len(intersection_coordinates[i][j]) > 0:
                    for intersection in intersection_coordinates[i][j]:
                        # Add a list of intersection coordinates
                        if intersection not in intersection_set:
                            intersection_set[intersection] = 0
                        # If the intersection is already in the set, increase the counter
                        else:
                            intersection_set[intersection] = (
                                intersection_set[intersection] + 1
                            )

    # 2. Implement the moves via N_independent_moves_path
    # 2.1 Reconstruct new move list regarding the parallel moves
    keep_running_flag = True
    count = 0
    # Why count < 5? Most of the path have less than 5 moves.
    while keep_running_flag and count < 5:
        keep_running_flag = True
        moves_in_scan = []
        destination_set = set()
        # 2.1.1 If there is no crossing path, implement one move for each path
        for path_in_moves in N_independent_moves_path:
            # Check if there are unimplemented moves in the path
            if len(path_in_moves) > 0:
                for move in path_in_moves:
                    crossing_path_flag = check_crossing_path(
                        matrix,
                        move[0],
                        intersection_set,
                        destination_set,
                        path_in_moves,
                    )
                    if not crossing_path_flag:
                        moves_in_scan.append(move[0])
                        path_in_moves.pop(0)
                        destination_set.add((move[0].to_row, move[0].to_col))
                    else:
                        break
        # 2.1.2 Parallelize the moves in the same round
        if len(moves_in_scan) > 0:
            moves_in_scan = regroup_parallel_moves(matrix, moves_in_scan)
            # 2.1.3 Implement the moves
            parallel_move_set.extend(moves_in_scan)
            for moves in moves_in_scan:
                matrix = move_atoms_noiseless(matrix, moves)
                for move in moves:
                    if (move.from_row, move.from_col) in intersection_set:
                        if intersection_set[(move.from_row, move.from_col)] > 0:
                            intersection_set[(move.from_row, move.from_col)] -= 1
                        else:
                            del intersection_set[(move.from_row, move.from_col)]

                    if (move.to_row, move.to_col) in intersection_set:
                        if intersection_set[(move.to_row, move.to_col)] > 0:
                            intersection_set[(move.to_row, move.to_col)] -= 1
                        else:
                            del intersection_set[(move.to_row, move.to_col)]
        else:
            keep_running_flag = False
        count += 1

    return matrix, parallel_move_set


def bfs_move_atom(grid, start, end, prev_path):
    n_rows, n_cols = grid.shape
    queue = deque([(start[0], start[1], [(start[0], start[1])])])
    visited = {(start[0], start[1])}

    while queue:
        current_row, current_col, path = queue.popleft()

        if (current_row, current_col) == end:
            if len(prev_path) > 0:
                prev_path = prev_path, path
            else:
                prev_path = path
            return prev_path, end

        len_path = len(path) - 1
        dr = (
            1
            if end[0] > path[len_path][0]
            else (-1 if end[0] < path[len_path][0] else 0)
        )
        dc = (
            1
            if end[1] > path[len_path][1]
            else (-1 if end[1] < path[len_path][1] else 0)
        )
        new_row, new_col = current_row + dr, current_col + dc

        in_bounds = (0 <= new_row < n_rows) and (0 <= new_col < n_cols)
        if (
            in_bounds
            and (new_row, new_col) not in visited
            and grid[new_row][new_col] == 0
        ):
            visited.add((new_row, new_col))
            queue.append((new_row, new_col, path + [(new_row, new_col)]))

    obstacle = (path[len_path][0] + dr, path[len_path][1] + dc)
    if obstacle == start:
        raise RuntimeError(
            f"bfs_move_atom failed to make progress from {start} toward {end}"
        )
    if not (0 <= obstacle[0] < n_rows and 0 <= obstacle[1] < n_cols):
        raise RuntimeError(
            f"bfs_move_atom stepped out of bounds: obstacle={obstacle}, end={end}"
        )

    if len(prev_path) > 0:
        prev_path = prev_path, path + [obstacle]
    else:
        prev_path = path + [obstacle]

    return prev_path, obstacle


##Find possible path between start and end position
def bfs_move_atom_og(grid, start, end, prev_path):
    queue = deque(
        [(start[0], start[1], [(start[0], start[1])])]
    )  # Use the queue to record current position and path
    visited = set()  # Record the visited positions
    visited.add((start[0], start[1]))

    # Start finding the path
    while queue:
        current_row, current_col, path = queue.popleft()  # Update current position

        # If we arrive end point, return the path
        if (current_row, current_col) == end:
            if len(prev_path) > 0:
                prev_path = prev_path, path
            else:
                prev_path = path

            return prev_path, end

        # Explore the next step (based on current position and end point)
        len_path = len(path) - 1
        dr = (
            1
            if end[0] > path[len_path][0]
            else (-1 if end[0] < path[len_path][0] else 0)
        )
        dc = (
            1
            if end[1] > path[len_path][1]
            else (-1 if end[1] < path[len_path][1] else 0)
        )
        new_row, new_col = current_row + dr, current_col + dc

        # Check if there is an obstacle there (If no, start from this new point to find next step)
        if (new_row, new_col) not in visited and grid[new_row][new_col] == 0:
            visited.add((new_row, new_col))
            queue.append((new_row, new_col, path + [(new_row, new_col)]))

    # If there is an obstacle on the path, we decompose the path: start->obstacle->target
    # Define the obstacle position
    obstacle = (path[len_path][0] + dr, path[len_path][1] + dc)

    # Update the move in path until obstacle
    if len(prev_path) > 0:
        prev_path = prev_path, path + [obstacle]
    else:
        prev_path = path + [obstacle]

    # [Path between start and obstacle] + [Path between obstacle and end]
    return prev_path, obstacle


def check_intersection(path1, path2):
    # Extract destination coordinates from both lists
    destinations1 = {(move[0].to_row, move[0].to_col) for move in path1}
    destinations1.add((path1[0][0].from_row, path1[0][0].from_col))
    destinations2 = {(move[0].to_row, move[0].to_col) for move in path2}
    destinations2.add((path2[0][0].from_row, path2[0][0].from_col))

    # Find intersections
    intersections = destinations1 & destinations2

    # Return result
    if intersections:
        return True, list(intersections)
    else:
        return False, []


def check_crossing_path(
    matrix, move, intersection_set, delay_destination, path_in_moves
):
    # Check if the destination is not in the intersection. If no intersection, implement the move
    if (move.to_row, move.to_col) not in intersection_set:
        return False
    # Check if the destination is end point of the path. If True, delay the move
    elif intersection_set[(move.to_row, move.to_col)] > 0 and (
        move.to_row,
        move.to_col,
    ) == (path_in_moves[-1][0].to_row, path_in_moves[-1][0].to_col):
        return True
    # If the destination is in the intersection set, but not passed yet this round, implement the move
    elif (move.to_row, move.to_col) not in delay_destination and matrix[move.to_row][
        move.to_col
    ] == 0:
        delay_destination.add((move.to_row, move.to_col))
        return False
    else:
        return True


def flatten_tuple(nested_tuple):
    # This function will flatten a nested tuple of lists into a single tuple of lists
    result = []

    def recursive_flatten(element):
        if isinstance(element, tuple):
            # If the element is a tuple, apply recursion to each item
            for item in element:
                recursive_flatten(item)
        elif isinstance(element, list):
            # If the element is a list, append it to the result
            result.append(tuple(element))

    # Start the recursion with the entire nested tuple
    recursive_flatten(nested_tuple)

    # Convert the list of tuples into a single tuple
    return tuple(result)

    # left_eject, bot_eject, top_eject, right_eject = [], [], [], []
    # len_x = len(matrix[0])
    # len_y = len(matrix)
    # for x in range(len_x):
    #     for y in range(len_y):
    #         if matrix[x][y] == 1 and target_config[x][y] == 0:
    #             if x >= y and x < len_y - y:
    #                 left_eject.append((x, y))
    #             elif x >= y and x >= len_y - y:
    #                 bot_eject.append((x, y))
    #             elif x < y and x <= len_y - y:
    #                 top_eject.append((x, y))
    #             elif x <= y and x >= len_y - y:
    #                 right_eject.append((x, y))
    # return left_eject, bot_eject, top_eject, right_eject


def generate_target_config(
    size: list,
    pattern: Configurations = 0,
    middle_size: list[int] | None = None,
    probability: float = 0.5,
) -> np.ndarray:
    """A function for generating common target configurations,
    such as checkerboard, zebra stripes, and middle fill.
    """
    array = np.zeros(size)
    if middle_size is None:
        middle_size = generate_middle_fifty(size[0])
    elif len(middle_size) == 0:
        middle_size = generate_middle_fifty(size[0])

    if pattern == Configurations.ZEBRA_HORIZONTAL:  # every other row
        for i in range(0, size[0], 2):
            array[i, :] = 1
    elif pattern == 1:  # every other col
        for i in range(0, size[1], 2):
            array[:, i] = 1
    elif pattern == 2:  # checkerboard
        array = np.indices(size).sum(axis=0) % 2
    elif pattern == 3:  # middle fill
        mrow = np.zeros([1, size[1]])
        mrow[
            0,
            int(size[1] / 2 - middle_size[1] / 2) : int(
                size[1] / 2 - middle_size[1] / 2
            )
            + middle_size[1],
        ] = 1
        for i in range(
            int(size[0] / 2 - middle_size[0] / 2),
            int(size[0] / 2 - middle_size[0] / 2) + middle_size[0],
        ):
            array[i, :] = mrow
    elif pattern == 4:
        for i in range(middle_size[0]):
            array[:, i] = 1
    elif pattern == 5:
        array = random_loading(size, probability=probability)
    return array


## helpers/wrappers


def _as_2d_occupancy(arr: np.ndarray, name: str) -> np.ndarray:
    """
    Normalize an occupancy array to a 2D single-species view.

    Parameters
    ----------
    arr : np.ndarray
        Occupancy array, expected to be either 2D or 3D with a singleton
        species axis.
    name : str
        Array name for error reporting.

    Returns
    -------
    np.ndarray
        2D occupancy view.

    Raises
    ------
    ValueError
        If the array cannot be interpreted as a 2D single-species occupancy
        grid.
    """
    arr = np.asarray(arr)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return arr[:, :, 0]
        raise ValueError(
            f"{name} has shape {arr.shape}; fast Hungarian helper expects a "
            "single-species grid, not multiple species."
        )

    raise ValueError(
        f"{name} has shape {arr.shape}; expected a 2D grid or 3D grid with "
        "singleton species axis."
    )
