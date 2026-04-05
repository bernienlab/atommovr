from __future__ import annotations

import copy
import numpy as np
from typing import Tuple, Literal

import atommovr.utils as movr
from atommovr.algorithms.source.ejection import ejection


def _int_sum(x: np.ndarray) -> int:
    """Return ``int(np.sum(x))`` with a signed accumulation dtype.

    Notes
    -----
    This avoids unsigned underflow/overflow bugs when subtracting counts that come
    from uint-typed occupancy arrays (e.g., ``np.uint8``).
    """
    return int(np.sum(x, dtype=np.int64))


def _as_2d_state(state: np.ndarray) -> np.ndarray:
    """
    Normalize BCv2 internal occupancy representation to a 2D (rows, cols) view.

    Why this exists
    ---------------
    BCv2 is logically single-species and most internal helpers reason about a
    2D occupancy grid. In the wider package, single-species matrices are often
    stored as (rows, cols, 1). This helper keeps the BCv2 internals robust to
    that representation without forcing the rest of the algorithm to care.
    """
    arr = np.asarray(state)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    raise ValueError(
        f"BCv2 expected 2D or (rows, cols, 1) single-species state; got shape {arr.shape}."
    )


def bcv2(array, do_ejection=False):
    if len(np.shape(array.matrix)) > 2 and np.shape(array.matrix)[2] == 2:
        raise ValueError(
            f"Atom array has shape {np.shape(array.matrix)}, which is not correct for single species. Did you meant to use a dual species algorithm?"
        )
    success_flag = False
    arr1 = copy.deepcopy(array)
    start_row, start_col, end_row, end_col = get_target_locs(arr1)
    # 1. prebalance (making sure target rows/cols have enough atoms)
    master_move_list, col_compact, success_flag = prebalance(arr1.matrix, arr1.target)
    _, _ = arr1.evaluate_moves(master_move_list)
    if success_flag:  # and col_compact == False:
        # _,_ = arr1.evaluate_moves(master_move_list)
        # 2. balance (distributing atoms between target rows according to needs)
        assignments = get_all_balance_assignments(start_row, end_row)
        for assignment in assignments:
            try:
                bal_moves = balance_rows(
                    arr1.matrix, arr1.target, assignment[0], assignment[1]
                )
                if assignment[0] != assignment[1] and len(bal_moves) > 0:
                    _, _ = arr1.evaluate_moves(bal_moves)
                    master_move_list.extend(bal_moves)
            except ValueError:
                return arr1.matrix, master_move_list, False

        # 3. compact
        com_moves = compact(arr1)
        if len(com_moves) > 0:
            _, _ = arr1.evaluate_moves(com_moves)
            master_move_list.extend(com_moves)

    if do_ejection:
        eject_moves, final_config = ejection(
            arr1.matrix,
            arr1.target,
            [0, len(arr1.matrix) - 1, 0, len(arr1.matrix[0]) - 1],
        )
        _, _ = arr1.evaluate_moves(eject_moves)
        master_move_list.extend(eject_moves)
        # 3.1 Check if the configuration is the same as the target configuration
        if np.array_equal(arr1.matrix, arr1.target.reshape(np.shape(arr1.matrix))):
            success_flag = True
    else:
        # 3.2 Check if the configuration (inside range of target) the same as the target configuration
        effective_config = np.multiply(
            arr1.matrix, arr1.target.reshape(np.shape(arr1.matrix))
        )
        if np.array_equal(effective_config, arr1.target.reshape(np.shape(arr1.matrix))):
            success_flag = True
    return arr1.matrix, master_move_list, success_flag


def special_case_algo_1d(init_config: np.ndarray, target_config: np.ndarray) -> "list":
    arr_copy = movr.AtomArray(np.shape(init_config)[:2])
    arr_copy.target = copy.deepcopy(target_config)
    arr_copy.matrix = copy.deepcopy(init_config)

    # first, find the column indices of the target sites
    # and those of the sites with atoms
    target_indices = np.where(arr_copy.target == 1)[1]
    atom_indices = np.where(arr_copy.matrix == 1)[1]

    if len(target_indices) != len(atom_indices):
        raise Exception(
            f"Number of atoms ({len(atom_indices)}) does not equal number of target sites ({len(target_indices)})."
        )

    # second, we can pair the atoms and make a list
    pairs = []
    for ind, target_index in enumerate(target_indices):
        atom_index = atom_indices[ind]
        pair = (target_index, atom_index)
        pairs.append(pair)
    # lastly, we can move atoms towards their target positions
    target_prepared = np.array_equal(arr_copy.target, arr_copy.matrix)
    move_set = []
    while not target_prepared:
        move_list = []
        for i, pair in enumerate(pairs):
            target_index, atom_index = pair
            if target_index != atom_index:
                new_atom_index = int(atom_index + np.sign(target_index - atom_index))
                move = movr.Move(0, atom_index, 0, new_atom_index)
                move_list.append(move)
                pairs[i] = (target_index, new_atom_index)
        if move_list != []:
            _, _ = arr_copy.evaluate_moves([move_list])
            move_set.append(move_list)
        else:
            break
    return move_set, atom_indices


# utility function that calculates the longest move distance between target sites and atom sites
def find_largest_dist_to_move(target_inds, atom_inds):
    if len(target_inds) > len(atom_inds):
        return np.inf
    max_dist = 0
    for ind, target_loc in enumerate(target_inds):
        atom_loc = atom_inds[ind]
        distance = np.abs(target_loc - atom_loc)
        if distance > max_dist:
            max_dist = distance
    return max_dist


def middle_fill_algo_1d(
    init_config: np.ndarray, target_config: np.ndarray
) -> Tuple[list, list]:
    arr_copy = movr.AtomArray(np.shape(init_config)[:2])
    arr_copy.target = copy.deepcopy(target_config)
    arr_copy.matrix = copy.deepcopy(init_config)
    # first, find the column indices of the target sites
    # and those of the sites with atoms
    target_indices = np.where(arr_copy.target == 1)[1]
    atom_indices = np.where(arr_copy.matrix == 1)[1]
    n_targets = len(target_indices)

    # second, find the optimal pairing of atoms if
    if n_targets == len(atom_indices):
        return special_case_algo_1d(init_config, target_config)
    elif n_targets > len(atom_indices):
        return [], []

    # third, find the centermost set of atoms
    avg_targ_pos = int(np.ceil(np.mean(target_indices)))
    count = 0
    sufficient_atoms = False
    while not sufficient_atoms:
        center_region = arr_copy.matrix[
            0, avg_targ_pos - count : avg_targ_pos + count + 1
        ]
        n_atoms_in_center_region = _int_sum(center_region)
        sufficient_atoms = n_targets <= n_atoms_in_center_region
        if not sufficient_atoms:
            count += 1
        else:
            break
    first_atom_loc = np.where(center_region == 1)[0][0] + avg_targ_pos - count

    # fourth, look to the adjacent sets and see if these are better
    look_right = True
    right_count = 0
    while look_right:
        list_ind = np.where(atom_indices == first_atom_loc)[0][0]
        current_r_atom_set = atom_indices[
            list_ind + right_count : list_ind + right_count + n_targets
        ]
        right_atom_set = atom_indices[
            list_ind + right_count + 1 : list_ind + right_count + n_targets + 1
        ]
        dist_r_current = find_largest_dist_to_move(target_indices, current_r_atom_set)
        dist_right = find_largest_dist_to_move(target_indices, right_atom_set)
        if dist_right > dist_r_current:
            look_right = False
        else:
            right_count += 1

    look_left = True
    left_count = 0
    while look_left:
        list_ind = np.where(atom_indices == first_atom_loc)[0][0]
        current_l_atom_set = atom_indices[
            list_ind - left_count : list_ind - left_count + n_targets
        ]
        left_atom_set = atom_indices[
            list_ind - left_count - 1 : list_ind - left_count + n_targets - 1
        ]
        dist_l_current = find_largest_dist_to_move(target_indices, current_l_atom_set)
        dist_left = find_largest_dist_to_move(target_indices, left_atom_set)
        if dist_left > dist_l_current:
            look_left = False
        else:
            left_count += 1

    if dist_l_current < dist_r_current:
        best_atom_set = current_l_atom_set
    else:
        best_atom_set = current_r_atom_set

    # fifth, find the best set and assign pairs
    pairs = []
    for ind, target_index in enumerate(target_indices):
        atom_index = best_atom_set[ind]
        pair = (target_index, atom_index)
        pairs.append(pair)

    # lastly, we can move atoms towards their target positions
    target_prepared = np.array_equal(arr_copy.target, arr_copy.matrix)
    move_set = []
    while not target_prepared:
        move_list = []
        for i, pair in enumerate(pairs):
            target_index, atom_index = pair
            if target_index != atom_index:
                new_atom_index = int(atom_index + np.sign(target_index - atom_index))
                move = movr.Move(0, atom_index, 0, new_atom_index)
                move_list.append(move)
                pairs[i] = (target_index, new_atom_index)
        if move_list != []:
            _, _ = arr_copy.evaluate_moves([move_list])
            move_set.append(move_list)
        else:
            target_prepared = True

    return move_set, best_atom_set


# Balance and Compact


def balance_rows(init_config: np.ndarray, target_config: np.ndarray, i: int, j: int):
    if i == j:
        return []
    n_rows_involved = j - i + 1
    m = i + (n_rows_involved // 2)
    n_req_top = _int_sum(target_config[i:m, :])
    n_atoms_top = _int_sum(init_config[i:m, :])
    n_req_bot = _int_sum(target_config[m : j + 1, :])
    n_atoms_bot = _int_sum(init_config[m : j + 1, :])
    diff_top = n_atoms_top - n_req_top
    diff_bot = n_atoms_bot - n_req_bot
    if (diff_top + diff_bot) < 0:
        raise ValueError(
            f"Insufficient number of atoms: deficit in rows {i}-{m-1} is {diff_top} and deficit in rows {m}-{j} is {diff_bot}."
        )

    current_state = init_config.copy()
    moves = []
    if diff_top < 0 and diff_bot > 0:
        n_to_move = min(-diff_top, diff_bot)
        current_state, round_moves = move_across_rows(
            current_state, n_to_move, i, j, m, -1
        )
        if len(round_moves) > 0:
            moves.extend(round_moves)
    elif diff_bot < 0 and diff_top > 0:
        n_to_move = min(-diff_bot, diff_top)
        current_state, round_moves = move_across_rows(
            current_state, n_to_move, i, j, m, 1
        )
        if len(round_moves) > 0:
            moves.extend(round_moves)
    else:
        n_to_move = 0
        pass
    # for debugging
    top_atoms = _int_sum(current_state[i:m, :])
    top_req = _int_sum(target_config[i:m, :])
    bot_atoms = _int_sum(current_state[m : j + 1, :])
    bot_req = _int_sum(target_config[m : j + 1, :])
    try:
        assert top_atoms >= top_req
        assert bot_atoms >= bot_req
    except AssertionError as err:
        raise RuntimeError(
            "move_across_rows failed to complete transfer:"
            f"Before: {n_atoms_top} on top, {n_atoms_bot} on bot, {n_to_move} to move."
            f"After: {top_atoms} on top, {bot_atoms} on bot."
            f"Needed: {top_req} on top, {bot_req} on bot."
        ) from err
    return moves


def _get_all_moves_btwn_rows_cols_checked(
    current_state: np.ndarray,
    from_row_ind: int,
    to_row_ind: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Bounds-checked wrapper for get_all_moves_btwn_rows_cols.

    Why this exists
    ---------------
    The original prebalance logic relied on IndexError to terminate deep searches.
    NumPy negative indexing wraps, so without explicit bounds checks the search can
    run forever for some directions/configurations.
    """
    n_rows: int = int(current_state.shape[0])
    if (
        from_row_ind < 0
        or from_row_ind >= n_rows
        or to_row_ind < 0
        or to_row_ind >= n_rows
    ):
        raise IndexError("row index out of bounds in prebalance search")
    return get_all_moves_btwn_rows_cols(current_state, from_row_ind, to_row_ind)


def _prebalance_above(
    current_state, start_row, end_row, n_targets, round_moves, direction
):
    n_movable_above = 0
    n_movable = 0
    row_offset = 0
    boundary_row = start_row if direction == -1 else end_row

    while n_movable_above == 0:
        if _int_sum(current_state) < n_targets:
            raise Exception("Insufficient atoms.")

        try:
            for off in range(row_offset + 1)[::-1]:
                src_row = boundary_row + (1 + off) * direction
                dst_row = boundary_row + off * direction

                from_cols, to_cols, n_movable = _get_all_moves_btwn_rows_cols_checked(
                    current_state, src_row, dst_row
                )

                # NOTE: keep the original condition exactly (no caching)
                if (
                    n_movable != 0
                    and _int_sum(current_state[start_row : end_row + 1, :]) < n_targets
                ):
                    above_moves = [
                        movr.Move(src_row, int(fc), dst_row, int(tc))
                        for fc, tc in zip(from_cols, to_cols, strict=True)
                    ]
                    current_state = movr.move_atoms_noiseless(
                        current_state, above_moves
                    )
                    round_moves.append(above_moves)
                else:
                    n_in_from_row = _int_sum(current_state[src_row, :])
                    if n_in_from_row > 0:
                        rows_in = 0
                        stuck_row = dst_row

                        while n_movable == 0:
                            for r_in in range(-1, rows_in)[::-1]:
                                src2 = stuck_row - r_in * direction
                                dst2 = stuck_row - (1 + r_in) * direction

                                f2, t2, n_sp_movable = (
                                    _get_all_moves_btwn_rows_cols_checked(
                                        current_state, src2, dst2
                                    )
                                )

                                if (
                                    n_sp_movable != 0
                                    and _int_sum(
                                        current_state[start_row : end_row + 1, :]
                                    )
                                    < n_targets
                                ):
                                    space_moves = [
                                        movr.Move(src2, int(fc), dst2, int(tc))
                                        for fc, tc in zip(f2, t2, strict=True)
                                    ]
                                    current_state = movr.move_atoms_noiseless(
                                        current_state, space_moves
                                    )
                                    round_moves.append(space_moves)

                                    f3, t3, n_movable = (
                                        _get_all_moves_btwn_rows_cols_checked(
                                            current_state,
                                            stuck_row,
                                            stuck_row - direction,
                                        )
                                    )
                                    if (
                                        _int_sum(
                                            current_state[start_row : end_row + 1, :]
                                        )
                                        < n_targets
                                        and n_movable != 0
                                    ):
                                        above_moves = [
                                            movr.Move(
                                                stuck_row,
                                                int(fc),
                                                stuck_row - direction,
                                                int(tc),
                                            )
                                            for fc, tc in zip(f3, t3, strict=True)
                                        ]
                                        current_state = movr.move_atoms_noiseless(
                                            current_state, above_moves
                                        )
                                        round_moves.append(above_moves)

                            rows_in += 1

            if n_movable > 0:
                n_movable_above = n_movable
            row_offset += 1

        except IndexError:
            row_offset += 1
            break

    return current_state, round_moves


def prebalance(init_config, target_config):
    success_flag = False

    # VECTORIZED SPEEDUP
    # Find the relevant rows and columns of the target configuration
    # row_max = 0
    # row_min = len(target_config)-1
    # col_max = 0
    # col_min = len(target_config[0])-1
    # for row in range(len(target_config)):
    #     for col in range(len(target_config[0])):
    #         if target_config[row,col] == 1:
    #             if row > row_max:
    #                 row_max = row
    #             if row < row_min:
    #                 row_min = row
    #             if col > col_max:
    #                 col_max = col
    #             if col < col_min:
    #                 col_min = col
    # start_row, start_col, end_row, end_col = row_min, col_min, row_max, col_max
    t = target_config
    if t.ndim == 3:
        t2 = t[:, :, 0]
    else:
        t2 = t
    rr, _ = np.where(t2 == 1)
    if rr.size == 0:
        return [], None, False
    start_row = int(rr.min())
    end_row = int(rr.max())
    # start_col = int(cc.min())
    # end_col = int(cc.max())

    n_atoms_row_region = _int_sum(init_config[start_row : end_row + 1, :])
    # n_atoms_col_region = _int_sum(init_config[:, start_col : end_col + 1])
    n_atoms_global = _int_sum(init_config)
    n_targets = _int_sum(target_config[start_row : end_row + 1, :])

    if n_atoms_global < n_targets:
        return [], None, success_flag

    # finding how many atoms we need to fill and generating moves
    n_to_fill_row = n_targets - n_atoms_row_region
    # n_to_fill_col = n_targets - n_atoms_col_region # linting error - unused

    moves = []
    if n_to_fill_row <= 0:
        col_compact = False
        success_flag = True
        return moves, col_compact, success_flag
    else:
        col_compact = False

        current_state = copy.deepcopy(init_config)
        while (
            np.sum(current_state[start_row : end_row + 1, :]) < n_targets
            and np.sum(current_state) >= n_targets
        ):
            round_moves = []
            # MOVING FROM ABOVE
            current_state, round_moves = _prebalance_above(
                current_state, start_row, end_row, n_targets, round_moves, -1
            )

            # MOVING FROM BELOW
            if np.sum(current_state[start_row : end_row + 1, :]) < n_targets:
                current_state, round_moves = _prebalance_above(
                    current_state, start_row, end_row, n_targets, round_moves, 1
                )

            moves.extend(round_moves)

        if np.sum(current_state[start_row : end_row + 1, :]) >= n_targets:
            success_flag = True
        return moves, col_compact, success_flag


def get_all_moves_btwn_rows_slow(init_config, from_row_ind, to_row_ind):
    if from_row_ind < 0 or to_row_ind < 0:
        raise IndexError

    from_row = init_config[from_row_ind, :]
    to_row = init_config[to_row_ind, :]

    available_source = np.flatnonzero(from_row == 1)

    # Boolean availability mask is faster than repeatedly shrinking an index array + np.isin
    free = (to_row == 0).copy()  # bool
    moves = []

    for atom_col in available_source:
        dest = None
        if atom_col - 1 >= 0 and free[atom_col - 1]:
            dest = atom_col - 1
        elif free[atom_col]:
            dest = atom_col
        elif atom_col + 1 < free.size and free[atom_col + 1]:
            dest = atom_col + 1

        if dest is not None:
            moves.append(movr.Move(from_row_ind, int(atom_col), to_row_ind, int(dest)))
            free[dest] = False

    return moves, len(moves)


def get_all_moves_btwn_rows_from_rows(
    from_row: np.ndarray,
    to_row: np.ndarray,
    from_row_ind: int,
    to_row_ind: int,
) -> tuple[list[movr.Move], int]:
    """
    Build a greedy, parallelizable move set that transfers atoms between two rows.

    Why this exists
    ---------------
    BCv2 calls row-to-row transfer many times. The dominant cost at high call counts
    is *Python overhead* (slicing, repeated attribute lookups, repeated bounds checks),
    not the simple local matching itself.

    This helper isolates the core logic so callers that already have the row slices
    can avoid reslicing `init_config` and can reuse temporary arrays in future
    optimizations.

    Contract
    --------
    - `from_row` and `to_row` are 1D integer arrays with occupancy in {0,1}.
    - The greedy policy matches the existing behavior:
        For each atom at column c (processed in increasing c),
        choose destination in priority order: c-1, c, c+1, subject to destination vacancy.
    - Each destination column is used at most once.

    Parameters
    ----------
    from_row, to_row
        1D occupancy arrays for source and destination rows (values 0/1).
    from_row_ind, to_row_ind
        Absolute row indices used to construct Move objects.

    Returns
    -------
    moves, n_moves
        Move list and its length.
    """
    if from_row_ind < 0 or to_row_ind < 0:
        raise IndexError

    # Fast exits
    if from_row.size == 0:
        return [], 0

    # `flatnonzero(from_row)` is equivalent to indices where from_row != 0
    src_cols = np.flatnonzero(from_row)
    if src_cols.size == 0:
        return [], 0

    # Free destination slots as a boolean array we can mutate.
    free = to_row == 0
    if not free.any():
        return [], 0

    n_cols = int(free.size)
    moves: list[movr.Move] = []
    append = moves.append  # localize for speed

    # Main greedy loop: minimal Python work per source.
    for c in src_cols:
        ci = int(c)

        left = ci - 1
        if left >= 0 and free[left]:
            append(movr.Move(from_row_ind, ci, to_row_ind, left))
            free[left] = False
            continue

        if free[ci]:
            append(movr.Move(from_row_ind, ci, to_row_ind, ci))
            free[ci] = False
            continue

        right = ci + 1
        if right < n_cols and free[right]:
            append(movr.Move(from_row_ind, ci, to_row_ind, right))
            free[right] = False
            continue

    return moves, len(moves)


def get_all_moves_btwn_rows_cols(
    init_config: np.ndarray,
    from_row_ind: int,
    to_row_ind: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Greedy row-to-row transfer, returning only (from_cols, to_cols).

    Why this exists
    ---------------
    BCv2 calls row-to-row transfer extremely frequently. Constructing `Move` objects
    in Python is expensive and often wasted when the caller only needs to know
    whether any moves exist.

    This function computes the same greedy matching policy as `get_all_moves_btwn_rows`
    but returns only integer arrays of column indices. Callers can then decide
    whether to materialize `Move` objects.

    Contract
    --------
    - `init_config` is a 3D occupancy array (rows, cols, 1) with values in {0,1}.
    - For each atom at source column c (processed in increasing c):
        choose destination priority: c-1, c, c+1, subject to vacancy in destination row.
      Each destination column is used at most once.

    Returns
    -------
    from_cols, to_cols
        1D arrays (dtype intp) of equal length.
    n_moves
        Number of matched moves.
    """
    if from_row_ind < 0 or to_row_ind < 0:
        raise IndexError

    if init_config.ndim != 3:
        raise ValueError(f"init_config must be 3D; got ndim={init_config.ndim}")

    from_row = init_config[from_row_ind, :, 0]
    to_row = init_config[to_row_ind, :, 0]

    out_from: list[int] = []
    out_to: list[int] = []

    src_cols = np.flatnonzero(from_row)
    if src_cols.size == 0:
        return out_from, out_to, 0

    free = to_row == 0
    if not free.any():
        # empty = np.zeros(0, dtype=np.intp) # linting error - unused
        return out_from, out_to, 0

    n_cols = int(free.size)

    # Collect into Python lists (cheap ints), then convert once at end.

    append_from = out_from.append
    append_to = out_to.append

    for c in src_cols:
        ci = int(c)

        left = ci - 1
        if left >= 0 and free[left]:
            append_from(ci)
            append_to(left)
            free[left] = False
            continue

        if free[ci]:
            append_from(ci)
            append_to(ci)
            free[ci] = False
            continue

        right = ci + 1
        if right < n_cols and free[right]:
            append_from(ci)
            append_to(right)
            free[right] = False
            continue

    if len(out_from) == 0:
        return out_from, out_to, 0

    from_cols = np.asarray(out_from, dtype=np.intp)
    to_cols = np.asarray(out_to, dtype=np.intp)
    return from_cols, to_cols, int(from_cols.size)


def get_all_moves_btwn_rows(
    init_config: np.ndarray,
    from_row_ind: int,
    to_row_ind: int,
) -> tuple[list[movr.Move], int]:
    """
    Backwards-compatible wrapper returning `Move` objects.

    Why this exists
    ---------------
    The rest of BCv2 expects a `list[Move]`. Internally we compute the same matching
    using `get_all_moves_btwn_rows_cols` and only construct `Move` objects once we
    know we have at least one move.
    """
    from_cols, to_cols, n_moves = get_all_moves_btwn_rows_cols(
        init_config, from_row_ind, to_row_ind
    )
    if n_moves == 0:
        return [], 0

    moves = [
        movr.Move(from_row_ind, int(fc), to_row_ind, int(tc))
        for fc, tc in zip(from_cols, to_cols, strict=True)
    ]
    return moves, n_moves


def get_all_moves_btwn_rows_faster(
    init_config: np.ndarray,
    from_row_ind: int,
    to_row_ind: int,
) -> tuple[list[movr.Move], int]:
    """
    Wrapper around `get_all_moves_btwn_rows_from_rows` that slices rows from `init_config`.

    Why this exists
    ---------------
    Maintains the existing BCv2 API, but routes through the optimized core routine.

    Parameters
    ----------
    init_config
        2D occupancy array (rows, cols) with values in {0,1}.
    from_row_ind, to_row_ind
        Row indices.

    Returns
    -------
    moves, n_moves
        Move list and its length.
    """
    if init_config.ndim != 3:
        raise ValueError(
            f"init_config must be 3D (rows, cols); got ndim={init_config.ndim}."
        )

    from_row = init_config[from_row_ind, :, 0]
    to_row = init_config[to_row_ind, :, 0]
    return get_all_moves_btwn_rows_from_rows(from_row, to_row, from_row_ind, to_row_ind)


def get_all_moves_btwn_cols(init_config, from_col_ind, to_col_ind):
    from_col = init_config[:, from_col_ind]
    to_col = init_config[:, to_col_ind, :]

    available_source = np.flatnonzero(from_col == 1)

    free = (to_col[:, 0] == 0).copy()  # bool
    moves = []

    for atom_row in available_source:
        dest = None
        if atom_row - 1 >= 0 and free[atom_row - 1]:
            dest = atom_row - 1
        elif free[atom_row]:
            dest = atom_row
        elif atom_row + 1 < free.size and free[atom_row + 1]:
            dest = atom_row + 1

        if dest is not None:
            moves.append(movr.Move(int(atom_row), from_col_ind, int(dest), to_col_ind))
            free[dest] = False

    return moves, len(moves)


def get_all_moves_btwn_rows_old(init_config, from_row_ind, to_row_ind):
    if from_row_ind < 0 or to_row_ind < 0:
        raise IndexError
    from_row = init_config[from_row_ind, :]
    to_row = init_config[to_row_ind, :]

    available_source = np.where(from_row == 1)[0]
    available_spots = np.where(to_row == 0)[0]

    moves = []
    for atom_col in available_source:
        move = None
        if atom_col - 1 in available_spots:
            move = movr.Move(from_row_ind, atom_col, to_row_ind, atom_col - 1)
            available_spots = available_spots[~np.isin(available_spots, atom_col - 1)]
        elif atom_col in available_spots:
            move = movr.Move(from_row_ind, atom_col, to_row_ind, atom_col)
            available_spots = available_spots[~np.isin(available_spots, atom_col)]
        elif atom_col + 1 in available_spots:
            move = movr.Move(from_row_ind, atom_col, to_row_ind, atom_col + 1)
            available_spots = available_spots[~np.isin(available_spots, atom_col + 1)]
        if move is not None:
            moves.append(move)
    n_atoms_movable = len(moves)
    return moves, n_atoms_movable


def get_all_moves_btwn_cols_old(init_config, from_col_ind, to_col_ind):
    from_col = init_config[:, from_col_ind]
    to_col = init_config[:, to_col_ind, :]

    available_source = np.where(from_col == 1)[0]
    available_spots = np.where(to_col == 0)[0]

    moves = []
    for atom_row in available_source:
        move = None
        if atom_row - 1 in available_spots:
            move = movr.Move(atom_row, from_col_ind, atom_row - 1, to_col_ind)
            available_spots = available_spots[~np.isin(available_spots, atom_row - 1)]
        elif atom_row in available_spots:
            move = movr.Move(atom_row, from_col_ind, atom_row, to_col_ind)
            available_spots = available_spots[~np.isin(available_spots, atom_row)]
        elif atom_row + 1 in available_spots:
            move = movr.Move(atom_row, from_col_ind, atom_row + 1, to_col_ind)
            available_spots = available_spots[~np.isin(available_spots, atom_row + 1)]
        if move is not None:
            moves.append(move)
    n_atoms_movable = len(moves)
    return moves, n_atoms_movable


def _apply_row_move(
    work_state: np.ndarray,
    from_row: int,
    to_row: int,
    cap: int | None = None,
) -> tuple[np.ndarray, list[movr.Move]]:
    """
    Execute movable atoms between two rows, optionally capped.

    This is the low-level primitive used for both support moves and transfer moves.
    """
    from_cols: np.ndarray
    to_cols: np.ndarray
    n_movable: int
    from_cols, to_cols, n_movable = _get_all_moves_btwn_rows_cols_checked(
        work_state,
        from_row,
        to_row,
    )

    if n_movable == 0:
        return work_state, []

    n_run: int = n_movable if cap is None else min(int(cap), int(n_movable))
    if n_run <= 0:
        return work_state, []

    moves: list[movr.Move] = [
        movr.Move(from_row, int(fc), to_row, int(tc))
        for fc, tc in zip(from_cols[:n_run], to_cols[:n_run], strict=True)
    ]
    new_state: np.ndarray = movr.move_atoms_noiseless(work_state, moves)
    return new_state, moves


def _try_direct_transfer(
    work_state: np.ndarray, remaining: int, boundary_src, boundary_dst
) -> tuple[np.ndarray, list[movr.Move]]:
    """Try to move atoms directly across the cut."""
    return _apply_row_move(
        work_state,
        boundary_src,
        boundary_dst,
        cap=remaining,
    )


def _try_clear_destination_side(
    work_state: np.ndarray, roi_rows
) -> tuple[np.ndarray, list[movr.Move]]:
    """
    Try one support move inside the destination ROI.

    We move atoms one step farther from the cut to create room near the cut.
    """
    roi_list: list[int] = list(roi_rows)
    for k in range(len(roi_list) - 1):
        from_row: int = roi_list[k]
        to_row: int = roi_list[k + 1]
        new_state, moves = _apply_row_move(work_state, from_row, to_row)
        if moves:
            return new_state, moves
    return work_state, []


def _try_pull_from_source_side(
    work_state: np.ndarray, source_rows
) -> tuple[np.ndarray, list[movr.Move]]:
    """
    Try one support move inside the source region.

    We move atoms one step closer to the cut to replenish the source boundary row.
    """
    src_list: list[int] = list(source_rows)
    for k in range(1, len(src_list)):
        from_row: int = src_list[k]
        to_row: int = src_list[k - 1]
        new_state, moves = _apply_row_move(work_state, from_row, to_row)
        if moves:
            return new_state, moves
    return work_state, []


def move_across_rows(
    current_state: np.ndarray,
    n_to_move: int,
    i: int,
    j: int,
    m: int,
    dir: Literal[-1, 1] = -1,
) -> tuple[np.ndarray, list[list[movr.Move]]]:
    """
    Move exactly ``n_to_move`` atoms across the cut between rows ``m - 1`` and ``m``.

    This helper is used by recursive row-balancing. Its job is narrower than a full
    rearrangement routine: it must realize a prescribed *net transfer* across a fixed
    partition, while preserving atom number and keeping an explicit move log.

    The key distinction is between:

    1. **transfer moves**
       Moves that actually cross the partition and therefore reduce the remaining
       transfer budget.

    2. **support moves**
       Moves wholly within the source side or wholly within the destination side that
       create room or bring atoms closer to the cut. These are allowed, but they do
       not count toward satisfying ``n_to_move``.

    The routine repeatedly tries to:
    - move atoms directly across the cut,
    - if blocked, clear space one step deeper into the destination ROI,
    - if starved, pull atoms one step closer from deeper in the source region.

    If a full pass produces no state change, the requested transfer is not being
    realized by the current routing logic, and the function raises instead of
    silently returning a partial result.

    Parameters
    ----------
    current_state : np.ndarray
        Binary occupancy array with shape ``(n_rows, n_cols, 1)`` or compatible.
    n_to_move : int
        Exact number of atoms to transfer across the cut.
    i : int
        Lowest row index in the active interval.
    j : int
        Highest row index in the active interval.
    m : int
        Partition index. The cut is between rows ``m - 1`` and ``m``.
    dir : {-1, 1}, default=-1
        Transfer direction. ``-1`` means move atoms from bottom half to top half.
        ``1`` means move atoms from top half to bottom half.

    Returns
    -------
    tuple[np.ndarray, list[list[movr.Move]]]
        Updated state and a list of executed moves.

    Raises
    ------
    ValueError
        If ``dir`` is not ``-1`` or ``1``.
    RuntimeError
        If the helper stalls before completing the requested transfer.
    Exception
        If the source side does not contain enough atoms to satisfy the request.
    """
    if dir not in (-1, 1):
        raise ValueError('Parameter "dir" must be -1 or 1.')

    if n_to_move < 0:
        raise ValueError("n_to_move must be nonnegative.")
    if n_to_move == 0:
        return current_state.copy(), []

    state: np.ndarray = current_state.copy()
    all_moves: list[list[movr.Move]] = []
    n_left_to_move: int = int(n_to_move)

    if dir == -1:
        # bottom -> top
        boundary_src: int = m
        boundary_dst: int = m - 1
        roi_rows: range = range(m - 1, i - 1, -1)  # destination side, from cut inward
        source_rows: range = range(m, j + 1)  # source side, from cut inward
        low_ind_source: int = m
        high_ind_source: int = j + 1
        top_slice = (slice(i, m), slice(None))
        bot_slice = (slice(m, j + 1), slice(None))
    else:
        # top -> bottom
        boundary_src = m - 1
        boundary_dst = m
        roi_rows = range(m, j + 1)  # destination side, from cut inward
        source_rows = range(m - 1, i - 1, -1)  # source side, from cut inward
        low_ind_source = i
        high_ind_source = m
        top_slice = (slice(i, m), slice(None))
        bot_slice = (slice(m, j + 1), slice(None))

    n_atoms_in_source: int = _int_sum(state[low_ind_source:high_ind_source])
    if n_atoms_in_source < n_to_move:
        raise Exception(
            f"Insufficient atoms. Only {n_atoms_in_source} in the source region; "
            f"requested transfer is {n_to_move}."
        )

    top_before: int = _int_sum(state[top_slice])
    bot_before: int = _int_sum(state[bot_slice])

    max_outer_iters: int = max(100, 4 * (j - i + 1) * state.shape[1])

    for _ in range(max_outer_iters):
        if n_left_to_move == 0:
            break

        state_before_pass: np.ndarray = state.copy()
        left_before_pass: int = n_left_to_move

        # Step 1: transfer as much as possible directly across the cut.
        state, transfer_moves = _try_direct_transfer(
            state, n_left_to_move, boundary_src=boundary_src, boundary_dst=boundary_dst
        )
        if transfer_moves:
            all_moves.append(transfer_moves)
            n_left_to_move -= len(transfer_moves)
            continue

        # Step 2: if blocked, try to create space near the destination boundary.
        state, clear_moves = _try_clear_destination_side(state, roi_rows=roi_rows)
        if clear_moves:
            all_moves.append(clear_moves)

            state, transfer_moves = _try_direct_transfer(
                state,
                n_left_to_move,
                boundary_src=boundary_src,
                boundary_dst=boundary_dst,
            )
            if transfer_moves:
                all_moves.append(transfer_moves)
                n_left_to_move -= len(transfer_moves)
                continue

        # Step 3: if still starved, try to pull atoms closer from the source side.
        state, pull_moves = _try_pull_from_source_side(state, source_rows)
        if pull_moves:
            all_moves.append(pull_moves)

            state, transfer_moves = _try_direct_transfer(
                state,
                n_left_to_move,
                boundary_src=boundary_src,
                boundary_dst=boundary_dst,
            )
            if transfer_moves:
                all_moves.append(transfer_moves)
                n_left_to_move -= len(transfer_moves)
                continue

        # Step 4: no real progress means the routing logic is jammed.
        state_changed: bool = not np.array_equal(state, state_before_pass)
        transfer_progress: bool = n_left_to_move < left_before_pass
        if not state_changed and not transfer_progress:
            break

    top_after: int = _int_sum(state[top_slice])
    bot_after: int = _int_sum(state[bot_slice])

    if dir == -1:
        moved: int = top_after - top_before
        expected_bot_delta: int = -n_to_move
        actual_bot_delta: int = bot_after - bot_before
        if moved != n_to_move or actual_bot_delta != expected_bot_delta:
            raise RuntimeError(
                "move_across_rows failed to complete requested bottom->top transfer. "
                f"Requested {n_to_move}, achieved {moved}. "
                f"Bottom delta was {actual_bot_delta}."
            )
    else:
        moved = bot_after - bot_before
        expected_top_delta: int = -n_to_move
        actual_top_delta: int = top_after - top_before
        if moved != n_to_move or actual_top_delta != expected_top_delta:
            raise RuntimeError(
                "move_across_rows failed to complete requested top->bottom transfer. "
                f"Requested {n_to_move}, achieved {moved}. "
                f"Top delta was {actual_top_delta}."
            )

    return state, all_moves


def move_across_rows_old(
    current_state: np.ndarray, n_to_move: int, i: int, j: int, m: int, dir=-1
):
    """
    Moves `n_to_move` atoms from row m to m-1 if dir = -1 or vice versa. If there aren't
    enough atoms, can access additional rows (subject to the constraint
    i < row and row < j).
    """

    round_moves = []  # master list of all moves taken in this procedure
    n_left_to_move = n_to_move

    ## specifying rows to move across and ROIs
    if dir == 1:
        start_row = m - 1
        end_row = m
        low_ind_roi = m
        high_ind_roi = j + 1
        low_ind_source = i
        high_ind_source = m
    elif dir == -1:
        start_row = m
        end_row = m - 1
        low_ind_roi = i
        high_ind_roi = m
        low_ind_source = m
        high_ind_source = j + 1
    else:
        raise ValueError('Parameter "dir" must be -1 or 1.')

    ## sanity check to make sure we have sufficient atoms
    n_atoms_in_source = _int_sum(current_state[low_ind_source:high_ind_source])
    n_atoms_in_roi = _int_sum(current_state[low_ind_roi:high_ind_roi])
    if n_atoms_in_source < n_to_move:
        raise Exception(
            f"Insufficient atoms. Only {n_atoms_in_source} in the source region (we need {n_to_move} more; only {n_atoms_in_roi} currently)."
        )

    ## continue looping until we move sufficient atoms.
    try_count = 0
    while n_left_to_move != 0 and try_count < 1000:
        try_count += 1
        n_movable_dir = 0
        row_offset = 0
        last_moves = [0]  # placeholder
        ## we loop until we are able to move atoms
        try_count2 = 0
        while n_movable_dir == 0 and try_count2 < 1000:
            try_count2 += 1
            try:
                move_set = []
                for off in range(row_offset + 1)[::-1]:
                    # across_move = 1 # linting error - unused
                    from_row = start_row + (off * dir)
                    to_row = end_row + (off * dir)
                    if i > from_row or i > to_row or j < from_row or j < to_row:
                        raise IndexError
                    # above_moves, n_movable = get_all_moves_btwn_rows(current_state,from_row, to_row)
                    from_cols, to_cols, n_movable = (
                        _get_all_moves_btwn_rows_cols_checked(
                            current_state, from_row, to_row
                        )
                    )

                    if (
                        n_movable != 0 and n_left_to_move != 0
                    ):  # check if there are atoms that can be moved, and if so move them
                        above_moves = [
                            movr.Move(from_row, int(fc), to_row, int(tc))
                            for fc, tc in zip(from_cols, to_cols, strict=True)
                        ]
                        if off == 0:
                            moves_to_run = above_moves[:n_left_to_move]
                        else:
                            moves_to_run = above_moves
                        current_state = movr.move_atoms_noiseless(
                            current_state, moves_to_run
                        )  # move_atoms
                        n_left_to_move -= len(moves_to_run)
                        move_set.append(moves_to_run)
                    else:  # if atoms CANNOT be moved
                        n_in_from_row = _int_sum(current_state[from_row, :])
                        ## Scenario 1: there are atoms to move, but no place to put them in the new row, so we have to clear room in ROI
                        if n_in_from_row > 0 and len(last_moves) > 0:
                            clear_space_in_roi_moves = []
                            rows_into_ROI = 0
                            while n_movable == 0:
                                stuck_row = start_row + dir * off
                                for r_in in range(rows_into_ROI + 1)[
                                    ::-1
                                ]:  # NKH change 05-09
                                    from_row = stuck_row + (1 + r_in) * dir
                                    to_row = stuck_row + (2 + r_in) * dir
                                    if (
                                        i > from_row
                                        or i > to_row
                                        or j < from_row
                                        or j < to_row
                                    ):
                                        raise IndexError
                                    # space_moves, n_sp_movable = get_all_moves_btwn_rows(current_state,from_row, to_row)
                                    from_sp_cols, to_sp_cols, n_sp_movable = (
                                        _get_all_moves_btwn_rows_cols_checked(
                                            current_state, from_row, to_row
                                        )
                                    )
                                    if (
                                        n_sp_movable != 0 and n_left_to_move != 0
                                    ):  # check if there are atoms that can be moved, and if so move them
                                        space_moves = [
                                            movr.Move(
                                                from_row, int(fcs), to_row, int(tcs)
                                            )
                                            for fcs, tcs in zip(
                                                from_sp_cols, to_sp_cols, strict=True
                                            )
                                        ]
                                        current_state = (
                                            movr.move_atoms_noiseless(  # move_atoms
                                                current_state, space_moves
                                            )
                                        )
                                        clear_space_in_roi_moves.append(space_moves)
                                        n_movable = n_sp_movable
                                rows_into_ROI += 1
                            if len(clear_space_in_roi_moves) > 0:
                                move_set.extend(clear_space_in_roi_moves)
                        ## Scenario 2: there are no atoms to move, so we have to take atoms from farther inside the source region
                        elif n_in_from_row == 0 or len(last_moves) == 0:
                            pull_atoms_from_reservoir_moves = []
                            rows_into_source = 0
                            while n_movable == 0:
                                stuck_row = start_row + dir * off
                                for r_in in range(-1, rows_into_source)[::-1]:
                                    from_row = stuck_row - (2 + r_in) * dir
                                    to_row = stuck_row - (1 + r_in) * dir
                                    if (
                                        i > from_row
                                        or i > to_row
                                        or j < from_row
                                        or j < to_row
                                    ):
                                        raise IndexError
                                    # space_moves, n_sp_movable = get_all_moves_btwn_rows(current_state,from_row, to_row)
                                    from_sp_cols, to_sp_cols, n_sp_movable = (
                                        _get_all_moves_btwn_rows_cols_checked(
                                            current_state, from_row, to_row
                                        )
                                    )
                                    if (
                                        n_sp_movable != 0 and n_left_to_move != 0
                                    ):  # check if there are atoms that can be moved, and if so move them
                                        space_moves = [
                                            movr.Move(
                                                from_row, int(fcs), to_row, int(tcs)
                                            )
                                            for fcs, tcs in zip(
                                                from_sp_cols, to_sp_cols, strict=True
                                            )
                                        ]
                                        current_state = (
                                            movr.move_atoms_noiseless(  # move_atoms
                                                current_state, space_moves
                                            )
                                        )
                                        pull_atoms_from_reservoir_moves.append(
                                            space_moves
                                        )
                                        n_movable = n_sp_movable
                                rows_into_source += 1
                            if len(pull_atoms_from_reservoir_moves) > 0:
                                move_set.extend(pull_atoms_from_reservoir_moves)
                # END DEBUG
                if len(move_set) > 0:
                    round_moves.extend(move_set)
                last_moves = move_set

                if n_movable > 0:
                    n_movable_dir = n_movable
                    break
                row_offset += 1
            except IndexError:
                row_offset += 1
                break

    return current_state, round_moves


def get_all_balance_assignments(start, end):
    assignments = []
    i = start
    j = end
    new_assignments = [(i, j)]
    n_a = len(new_assignments)
    while n_a > 0:
        assignment_list = []
        for assignment in new_assignments:
            i = assignment[0]
            j = assignment[1]
            next_layer = get_next_balance_assignment(i, j)
            assignment_list.extend(next_layer)
        assignments.extend(new_assignments)
        if len(assignment_list) > 0:
            new_assignments = assignment_list
        else:
            break
    return assignments


def get_next_balance_assignment(i, j):
    n_rows_involved = j - i + 1
    m = i + (n_rows_involved // 2)
    next_list = []
    if i != j and i < j:
        next_list.append((i, m - 1))
        next_list.append((m, j))
    return next_list


def get_target_locs(array):
    """
    Return the bounding box (start_row, start_col, end_row, end_col) of the target region.

    Notes
    -----
    Vectorized implementation to avoid O(H*W) Python loops.
    """
    targ = array.target
    if targ.ndim == 3:
        targ2 = targ[:, :, 0]
    else:
        targ2 = targ

    rr, cc = np.where(targ2 == 1)
    if rr.size == 0:
        # No target sites: treat as empty region
        return 0, 0, -1, -1

    return int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())


def get_target_locs_old(
    array,
):  # Find the relevant rows and columns of the target configuration
    """
    Finds the boundaries of array.target (i.e. the biggest square which contains all atoms in the target config).DS_Store

    ## Parameters:
        array : AtomArray()

    ## Returns:
        start_row : int
            the index of the first row where there are atoms in the target config
        start_col : int
            the index of the first column where there are atoms in the target config
        end_row : int
            the index of the last row where there are atoms in the target config
        end_col : int
            the index of the last column where there are atoms in the target config
    """
    n_rows = len(array.target)
    n_cols = len(array.target[0])
    row_max = int(0)
    row_min = int(n_rows - 1)
    col_max = int(0)
    col_min = int(n_cols - 1)
    for row in range(n_rows):
        for col in range(n_cols):
            if array.target[row, col, 0] == 1:
                if row > row_max:
                    row_max = row
                if row < row_min:
                    row_min = row
                if col > col_max:
                    col_max = col
                if col < col_min:
                    col_min = col
    start_row, start_col, end_row, end_col = row_min, col_min, row_max, col_max
    return start_row, start_col, end_row, end_col


def compact(array) -> list[list[movr.Move]]:
    """
    Compact atoms horizontally into the target rectangle using legal shared AOD frames.

    Why this exists
    ---------------
    Under the updated collision model, the fully symmetric inward crunch is not a
    physically admissible shared AOD command pattern: near the condensation column,
    inward tones from both sides can create colliding tweezers / pile-ups.

    This implementation compares the two admissible "almost symmetric" templates:
    one with the left center-closing tone removed and one with the right
    center-closing tone removed. For each template, only rows with positive vote
    and no induced collision are activated. The higher-scoring legal template is
    applied each round.

    Parameters
    ----------
    array
        Atom array whose occupancy is to be compacted.

    Returns
    -------
    list of list of Move
        Sequence of applied parallel move rounds.
    """
    arr1 = copy.deepcopy(array)

    start_row, start_col, end_row, end_col = get_target_locs(arr1)
    n_rows: int = len(arr1.target)
    n_cols: int = len(arr1.target[0])

    global_move_set: list[list[movr.Move]] = []
    while True:
        col_counts: np.ndarray = np.sum(
            arr1.matrix[start_row : end_row + 1, start_col : end_col + 1, 0],
            axis=0,
            dtype=np.int64,
        )
        min_col_ind: int = int(start_col + int(np.argmin(col_counts)))

        n_target_rows: int = end_row - start_row + 1
        r_vote_tally: np.ndarray = np.zeros(n_target_rows, dtype=np.float64)
        l_vote_tally: np.ndarray = np.zeros(n_target_rows, dtype=np.float64)
        move_arr: np.ndarray = np.zeros((n_target_rows, 2), dtype=object)

        for i, row in enumerate(range(start_row, end_row + 1)):
            atom_in_row: int = int(arr1.matrix[row, min_col_ind, 0])
            if atom_in_row != 0:
                r_vote_tally[i] = -np.e
                l_vote_tally[i] = -np.e

            move_set, best_atom_set = middle_fill_algo_1d(
                arr1.matrix[row, :, :].reshape(1, n_cols, 1),
                arr1.target[row, :, :].reshape(1, n_cols, 1),
            )

            round0: list[movr.Move] = (
                move_set[0]
                if (isinstance(move_set, list) and len(move_set) > 0)
                else []
            )
            round0_edges: set[tuple[int, int]] = {
                (int(m.from_col), int(m.to_col)) for m in round0
            }
            move_arr[i, 0] = best_atom_set
            move_arr[i, 1] = round0_edges

        for col in range(n_cols):
            move_dir: int = int(np.sign(min_col_ind - col))
            if move_dir == -1:
                for i, row in enumerate(range(start_row, end_row + 1)):
                    cond1: bool = bool(r_vote_tally[i] != -np.e)
                    cond2: bool = bool(int(arr1.matrix[row, col, 0]) == 1)
                    cond3: bool = bool(col in move_arr[i, 0])
                    if cond1 and cond2 and cond3:
                        vote: int = int((int(col), int(col - 1)) in move_arr[i, 1])
                        r_vote_tally[i] += -1 + 2 * vote
            elif move_dir == 1:
                for i, row in enumerate(range(start_row, end_row + 1)):
                    cond1 = bool(l_vote_tally[i] != -np.e)
                    cond2 = bool(int(arr1.matrix[row, col, 0]) == 1)
                    cond3 = bool(col in move_arr[i, 0])
                    if cond1 and cond2 and cond3:
                        vote = int((int(col), int(col + 1)) in move_arr[i, 1])
                        l_vote_tally[i] += -1 + 2 * vote

        total_vote_tally: np.ndarray = r_vote_tally + l_vote_tally

        # Direction-only fallbacks retained from the original implementation.
        r_comh_AOD_cmds: np.ndarray = np.zeros(n_cols, dtype=np.uint8)
        l_comh_AOD_cmds: np.ndarray = np.zeros(n_cols, dtype=np.uint8)
        r_comv_AOD_cmds: np.ndarray = np.zeros(n_rows, dtype=np.uint8)
        l_comv_AOD_cmds: np.ndarray = np.zeros(n_rows, dtype=np.uint8)
        r_vote_sum: float = 0.0
        l_vote_sum: float = 0.0

        for row_ind in range(n_target_rows):
            n_r_votes: float = float(r_vote_tally[row_ind])
            n_l_votes: float = float(l_vote_tally[row_ind])
            abs_row: int = row_ind + start_row
            if n_r_votes > 0:
                r_comv_AOD_cmds[abs_row] = np.uint8(1)
                r_vote_sum += n_r_votes
            elif n_l_votes > 0:
                l_comv_AOD_cmds[abs_row] = np.uint8(1)
                l_vote_sum += n_l_votes

        if min_col_ind + 1 < n_cols:
            r_comh_AOD_cmds[min_col_ind + 1 :] = np.uint8(3)
        if min_col_ind > 0:
            l_comh_AOD_cmds[:min_col_ind] = np.uint8(2)

        # Two legal near-symmetric crunch templates:
        #   left-deleted:  remove (c-1) -> c  by zeroing source column c-1
        #   right-deleted: remove (c+1) -> c  by zeroing source column c+1
        left_del_comh_AOD_cmds: np.ndarray = np.zeros(n_cols, dtype=np.uint8)
        right_del_comh_AOD_cmds: np.ndarray = np.zeros(n_cols, dtype=np.uint8)
        left_del_comv_AOD_cmds: np.ndarray = np.zeros(n_rows, dtype=np.uint8)
        right_del_comv_AOD_cmds: np.ndarray = np.zeros(n_rows, dtype=np.uint8)

        if min_col_ind > 0:
            left_del_comh_AOD_cmds[:min_col_ind] = np.uint8(2)
            right_del_comh_AOD_cmds[:min_col_ind] = np.uint8(2)
        if min_col_ind + 1 < n_cols:
            left_del_comh_AOD_cmds[min_col_ind + 1 :] = np.uint8(3)
            right_del_comh_AOD_cmds[min_col_ind + 1 :] = np.uint8(3)

        if min_col_ind - 1 >= 0:
            left_del_comh_AOD_cmds[min_col_ind - 1] = np.uint8(0)
        if min_col_ind + 1 < n_cols:
            right_del_comh_AOD_cmds[min_col_ind + 1] = np.uint8(0)

        left_deleted_collision_mask: np.ndarray = np.zeros(n_rows, dtype=np.bool_)
        right_deleted_collision_mask: np.ndarray = np.zeros(n_rows, dtype=np.bool_)

        # Left-deleted collisions:
        #   (c-2 -> c-1) collides with occupied c-1
        #   (c+1 -> c)   collides with occupied c
        if min_col_ind - 2 >= 0:
            left_deleted_collision_mask |= arr1.matrix[:, min_col_ind - 1, 0].astype(
                bool
            ) & arr1.matrix[:, min_col_ind - 2, 0].astype(bool)
        if min_col_ind + 1 < n_cols:
            left_deleted_collision_mask |= arr1.matrix[:, min_col_ind + 1, 0].astype(
                bool
            ) & arr1.matrix[:, min_col_ind, 0].astype(bool)

        # Right-deleted collisions:
        #   (c+2 -> c+1) collides with occupied c+1
        #   (c-1 -> c)   collides with occupied c
        if min_col_ind + 2 < n_cols:
            right_deleted_collision_mask |= arr1.matrix[:, min_col_ind + 1, 0].astype(
                bool
            ) & arr1.matrix[:, min_col_ind + 2, 0].astype(bool)
        if min_col_ind - 1 >= 0:
            right_deleted_collision_mask |= arr1.matrix[:, min_col_ind - 1, 0].astype(
                bool
            ) & arr1.matrix[:, min_col_ind, 0].astype(bool)

        left_deleted_vote_sum: float = 0.0
        right_deleted_vote_sum: float = 0.0

        for row_ind in range(n_target_rows):
            abs_row = row_ind + start_row
            row_vote: float = float(total_vote_tally[row_ind])
            if row_vote <= 0:
                continue

            if not bool(left_deleted_collision_mask[abs_row]):
                left_del_comv_AOD_cmds[abs_row] = np.uint8(1)
                left_deleted_vote_sum += row_vote

            if not bool(right_deleted_collision_mask[abs_row]):
                right_del_comv_AOD_cmds[abs_row] = np.uint8(1)
                right_deleted_vote_sum += row_vote

        left_deleted_moves: list[movr.Move] = movr.get_move_list_from_AOD_cmds(
            left_del_comh_AOD_cmds,
            left_del_comv_AOD_cmds,
        )
        right_deleted_moves: list[movr.Move] = movr.get_move_list_from_AOD_cmds(
            right_del_comh_AOD_cmds,
            right_del_comv_AOD_cmds,
        )
        r_moves: list[movr.Move] = movr.get_move_list_from_AOD_cmds(
            r_comh_AOD_cmds,
            r_comv_AOD_cmds,
        )
        l_moves: list[movr.Move] = movr.get_move_list_from_AOD_cmds(
            l_comh_AOD_cmds,
            l_comv_AOD_cmds,
        )

        moves_options: list[list[movr.Move]] = [
            left_deleted_moves,
            right_deleted_moves,
            r_moves,
            l_moves,
        ]
        vote_sums: list[float] = [
            left_deleted_vote_sum,
            right_deleted_vote_sum,
            r_vote_sum,
            l_vote_sum,
        ]

        best_ind: int = int(np.argmax(vote_sums))
        move_list: list[movr.Move] = moves_options[best_ind]

        if move_list != []:
            arr1.move_atoms(move_list)
            global_move_set.append(move_list)
        else:
            break

    return global_move_set


def compact_old(array):
    arr1 = copy.deepcopy(array)

    start_row, start_col, end_row, end_col = get_target_locs(arr1)
    n_rows = len(arr1.target)
    n_cols = len(arr1.target[0])

    global_move_set = []
    while True:
        """
        1. Loop through columns in target config and count how many atoms there are.
        2. Select the column with the least number of atoms.
        3. For all unoccupied rows in this column:
            i. Count the number of atoms that want to move from their current position towards the selected column.
            ii. Count the number of atoms that do NOT want to move.
            iii. If the number of atoms that want to move is greater than the number of atoms that do NOT want to move, add the row to row_list
        4. Condense all atoms in rows in rows_list inwards."""

        # counting how many vacancies are in columns
        # SPEEDUP
        # col_ns = []
        # for col in range(start_col, end_col+1):
        #     n_in_col = _int_sum(arr1.matrix[start_row:end_row+1,col, 0])
        #     col_ns.append(n_in_col)
        # min_n_col = min(col_ns)
        # min_col_ind = col_ns.index(min_n_col) + start_col #np.where(col_ns == min_n_col)[0][0]+start_col
        col_counts = np.sum(
            arr1.matrix[start_row : end_row + 1, start_col : end_col + 1, 0],
            axis=0,
            dtype=np.int64,
        )
        min_col_ind = int(start_col + int(np.argmin(col_counts)))

        r_vote_tally = np.zeros(len(range(start_row, end_row + 1)))
        l_vote_tally = np.zeros(len(range(start_row, end_row + 1)))
        move_arr = np.zeros([end_row - start_row + 1, 2], dtype="object")
        for i, row in enumerate(range(start_row, end_row + 1)):
            atom_in_row = arr1.matrix[row, min_col_ind, 0]
            if atom_in_row != 0:
                r_vote_tally[i] = -np.e  # code for automatic no vote
                l_vote_tally[i] = -np.e
            move_set, best_atom_set = middle_fill_algo_1d(
                arr1.matrix[row, :, :].reshape(1, len(arr1.target[0]), 1),
                arr1.target[row, :, :].reshape(1, len(arr1.target[0]), 1),
            )
            # move_arr[i, 1] = move_set
            # move_arr[i, 0] = best_atom_set
            # SPEEDUP Cache membership as tuples to avoid O(n) Move.__eq__ scans
            # move_set is a list-of-rounds; we only query round 0 in your voting.
            round0 = (
                move_set[0]
                if (isinstance(move_set, list) and len(move_set) > 0)
                else []
            )
            right_edges = set((int(m.from_col), int(m.to_col)) for m in round0)
            move_arr[i, 1] = right_edges
            move_arr[i, 0] = best_atom_set
        for col in range(n_cols):
            move_dir = np.sign(min_col_ind - col)
            if move_dir == -1:
                for i, row in enumerate(range(start_row, end_row + 1)):
                    cond1 = r_vote_tally[i] != -np.e
                    cond2 = int(arr1.matrix[row, col, 0]) == 1
                    cond3 = col in move_arr[i, 0]
                    if cond1 and cond2 and cond3:
                        vote = int(
                            (int(col), int(col - 1)) in move_arr[i, 1]
                        )  # SPEEDUP int(movr.Move(0, col, 0, col-1) in move_arr[i,1][0])
                        r_vote_tally[i] += -1 + 2 * vote
            elif move_dir == 1:
                for i, row in enumerate(range(start_row, end_row + 1)):
                    cond1 = l_vote_tally[i] != -np.e
                    cond2 = int(arr1.matrix[row, col, 0]) == 1
                    cond3 = col in move_arr[i, 0]
                    if cond1 and cond2 and cond3:
                        vote = int(
                            (int(col), int(col + 1)) in move_arr[i, 1]
                        )  # SPEEDUP int(movr.Move(0, col, 0, col+1) in move_arr[i,1][0])
                        l_vote_tally[i] += -1 + 2 * vote

        comh_AOD_cmds = np.zeros(n_cols, dtype=np.uint8)
        comv_AOD_cmds = np.zeros(n_rows, dtype=np.uint8)
        total_vote_sum = 0

        collision_mask = np.zeros(n_rows, dtype=np.bool_)
        if 0 < min_col_ind < (n_cols - 1):
            collisions = (
                arr1.matrix[:, min_col_ind - 1, 0] & arr1.matrix[:, min_col_ind + 1, 0]
            )
            collision_mask = collisions == 1
        # SPEEDUP
        # collision_inds = [] # FIX
        # # checking for collisions in the center column around which we condense
        # if min_col_ind not in [0,n_cols-1]:
        #     collisions = arr1.matrix[:,min_col_ind-1,0]*arr1.matrix[:,min_col_ind+1,0]
        #     if _int_sum(collisions) > 0:
        #         collision_inds = np.where(collisions == 1)[0]

        for row_ind in range(len(r_vote_tally)):
            if r_vote_tally[row_ind] + l_vote_tally[row_ind] > 0 and (
                not collision_mask[row_ind + start_row]
            ):  # SPEEDUP row_ind+start_row not in collision_inds:
                comv_AOD_cmds[row_ind + start_row] = np.uint8(1)
                total_vote_sum += r_vote_tally[row_ind] + l_vote_tally[row_ind]

        # for col_ind in range(n_cols):
        #     if np.sign(col_ind-min_col_ind) == 1:
        #         vert_AOD_cmds[col_ind] = np.uint8(3)
        #     elif np.sign(col_ind-min_col_ind) == -1:
        #         vert_AOD_cmds[col_ind] = np.uint8(2)
        # SPEEDUP
        if min_col_ind > 0:
            comh_AOD_cmds[:min_col_ind] = np.uint8(2)
        if min_col_ind + 1 < n_cols:
            comh_AOD_cmds[min_col_ind + 1 :] = np.uint8(3)

        r_comh_AOD_cmds = np.zeros(n_cols, dtype=np.uint8)
        l_comh_AOD_cmds = np.zeros(n_cols, dtype=np.uint8)
        r_comv_AOD_cmds = np.zeros(n_rows, dtype=np.uint8)
        l_comv_AOD_cmds = np.zeros(n_rows, dtype=np.uint8)
        r_vote_sum = 0
        l_vote_sum = 0
        for row_ind in range(len(r_vote_tally)):
            n_r_votes = r_vote_tally[row_ind]
            n_l_votes = l_vote_tally[row_ind]
            if n_r_votes > 0:
                r_comv_AOD_cmds[row_ind + start_row] = np.uint8(1)
                r_vote_sum += n_r_votes
            elif n_l_votes > 0:
                l_comv_AOD_cmds[row_ind + start_row] = np.uint8(1)
                l_vote_sum += n_l_votes
        # for col_ind in range(min_col_ind+1, n_cols):
        #     r_vert_AOD_cmds[col_ind] = np.uint8(3)
        # for col_ind in range(0,min_col_ind):
        #     l_vert_AOD_cmds[col_ind] = np.uint8(2)
        # SPEEDUP
        if min_col_ind + 1 < n_cols:
            r_comh_AOD_cmds[min_col_ind + 1 :] = np.uint8(3)
        if min_col_ind > 0:
            l_comh_AOD_cmds[:min_col_ind] = np.uint8(2)

        crunch_moves = movr.get_move_list_from_AOD_cmds(comh_AOD_cmds, comv_AOD_cmds)
        r_moves = movr.get_move_list_from_AOD_cmds(r_comh_AOD_cmds, r_comv_AOD_cmds)
        l_moves = movr.get_move_list_from_AOD_cmds(l_comh_AOD_cmds, l_comv_AOD_cmds)
        moves_options = [crunch_moves, r_moves, l_moves]
        vote_sums = [total_vote_sum, r_vote_sum, l_vote_sum]
        move_list = moves_options[np.argmax(vote_sums)]
        if move_list != []:
            # before = arr1.matrix.copy()
            arr1.move_atoms(move_list)
            # if np.array_equal(arr1.matrix, before):
            #     raise RuntimeError("compact(): applied moves but matrix did not change.")
            global_move_set.append(move_list)
        else:
            break

    return global_move_set
