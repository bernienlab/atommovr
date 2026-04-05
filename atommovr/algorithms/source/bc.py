## Modified Balance and Compact algorithm
from atommovr.utils.move_utils import Move, move_atoms

import numpy as np
import copy

from atommovr.utils.Move import Move

# Balance and Compact


def balance_rows(init_config: np.ndarray, target_config: np.ndarray, i: int, j: int):
    if i == j:
        return []
    difference = j - i + 1
    m = i + (difference // 2)
    n_req_top = np.sum(target_config[i:m, :])
    n_atoms_top = np.sum(init_config[i:m, :])
    n_req_bot = np.sum(target_config[m : j + 1, :])
    n_atoms_bot = np.sum(init_config[m : j + 1, :])
    diff_top = n_atoms_top - n_req_top
    diff_bot = n_atoms_bot - n_req_bot
    if (diff_top + diff_bot) < 0:
        raise ValueError(
            f"Insufficient number of atoms: deficit in rows {i}-{m-1} is {diff_top} and deficit in rows {m}-{j} is {diff_bot}."
        )
    current_state = copy.deepcopy(init_config)
    moves = []
    n_to_move = int(np.floor(np.abs(diff_bot - diff_top) / 2))
    if diff_bot == diff_top:
        pass
    elif diff_top < diff_bot:
        current_state, round_moves = move_across_rows(
            current_state, n_to_move, i, j, m, -1
        )
        if len(round_moves) > 0:
            moves.extend(round_moves)
    elif diff_bot < diff_top:
        current_state, round_moves = move_across_rows(
            current_state, n_to_move, i, j, m, 1
        )
        if len(round_moves) > 0:
            moves.extend(round_moves)
    return moves


def prebalance(init_config, target_config):
    success_flag = False

    # Find the relevant rows and columns of the target configuration
    row_max = 0
    row_min = len(target_config) - 1
    col_max = 0
    col_min = len(target_config[0]) - 1
    for row in range(len(target_config)):
        for col in range(len(target_config[0])):
            if target_config[row, col] == 1:
                if row > row_max:
                    row_max = row
                if row < row_min:
                    row_min = row
                if col > col_max:
                    col_max = col
                if col < col_min:
                    col_min = col
    start_row, start_col, end_row, end_col = row_min, col_min, row_max, col_max

    n_atoms_row_region = np.sum(init_config[start_row : end_row + 1, :])
    n_atoms_col_region = np.sum(init_config[:, start_col : end_col + 1])
    n_atoms_global = np.sum(init_config)
    n_targets = np.sum(target_config[start_row : end_row + 1, :])

    if n_atoms_global < n_targets:
        return [], None, success_flag

    # finding how many atoms we need to fill and generating moves
    n_to_fill_row = n_targets - n_atoms_row_region
    n_to_fill_col = n_targets - n_atoms_col_region

    moves = []
    if n_to_fill_row <= 0:
        col_compact = False
        success_flag = True
        return moves, col_compact, success_flag
    elif n_to_fill_col <= 0:
        col_compact = True
        success_flag = True
        return moves, col_compact, success_flag
    elif n_to_fill_col >= n_to_fill_row:
        col_compact = False

        current_state = copy.deepcopy(init_config)
        while np.sum(current_state[start_row : end_row + 1, :]) < n_targets:
            round_moves = []

            # MOVING FROM ABOVE
            n_movable_above = 0
            row_offset = 0
            while n_movable_above == 0:
                try:
                    move_set = []
                    for off in range(row_offset + 1)[::-1]:
                        above_moves, n_movable = get_all_moves_btwn_rows(
                            current_state, start_row - 1 - off, start_row - off
                        )
                        if (
                            n_movable != 0
                            and np.sum(current_state[start_row : end_row + 1, :])
                            < n_targets
                        ):  # check if there are atoms that can be moved, and if so move them
                            current_state, _ = move_atoms(current_state, above_moves)
                        else:  # if no atoms can be moved, figure out why
                            n_in_from_col = np.sum(
                                current_state[start_row - 1 - off, :]
                            )
                            if (
                                n_in_from_col > 0
                            ):  # if there are no spots for new atoms to come, make space by pushing atoms farther inside
                                rows_in = 0
                                while n_movable == 0:
                                    stuck_row = start_row - off
                                    for r_in in range(rows_in + 1)[::-1]:
                                        space_moves, n_sp_movable = (
                                            get_all_moves_btwn_rows(
                                                current_state,
                                                stuck_row + 1 + r_in,
                                                stuck_row + 2 + r_in,
                                            )
                                        )
                                        if (
                                            n_sp_movable != 0
                                            and np.sum(
                                                current_state[
                                                    start_row : end_row + 1, :
                                                ]
                                            )
                                            < n_targets
                                        ):  # check if there are atoms that can be moved, and if so move them
                                            current_state, _ = move_atoms(
                                                current_state, space_moves
                                            )
                                            move_set.append(space_moves)
                                            above_moves, n_movable = (
                                                get_all_moves_btwn_rows(
                                                    current_state,
                                                    stuck_row,
                                                    stuck_row + 1,
                                                )
                                            )
                                    rows_in += 1
                                if (
                                    np.sum(current_state[start_row : end_row + 1, :])
                                    < n_targets
                                ):
                                    current_state, _ = move_atoms(
                                        current_state, above_moves
                                    )
                        if len(above_moves) > 0:
                            move_set.append(above_moves)
                    if n_movable > 0:
                        n_movable_above = n_movable
                    row_offset += 1
                    if len(move_set) > 0:
                        round_moves.extend(move_set)
                except IndexError:
                    row_offset += 1
                    break

            # MOVING FROM BELOW
            if np.sum(current_state[start_row : end_row + 1, :]) < n_targets:

                # get atoms from below
                n_movable_below = 0
                row_offset = 0
                while n_movable_below == 0:
                    try:
                        move_set = []
                        for off in range(row_offset + 1)[::-1]:
                            below_moves, n_movable = get_all_moves_btwn_rows(
                                current_state, end_row + 1 + off, end_row + off
                            )
                            if (
                                n_movable != 0
                                and np.sum(current_state[start_row : end_row + 1, :])
                                < n_targets
                            ):  # check if there are atoms that can be moved, and if so move them
                                current_state, _ = move_atoms(
                                    current_state, below_moves
                                )
                            else:  # if no atoms can be moved, figure out why
                                n_in_from_col = np.sum(
                                    current_state[end_row + 1 + off, :]
                                )
                                if (
                                    n_in_from_col > 0
                                ):  # if there are no spots for new atoms to come, make space by pushing atoms farther inside
                                    rows_in = 0
                                    while n_movable == 0:
                                        stuck_row = end_row + off
                                        for r_in in range(rows_in + 1)[::-1]:
                                            space_moves, n_sp_movable = (
                                                get_all_moves_btwn_rows(
                                                    current_state,
                                                    stuck_row - 1 - r_in,
                                                    stuck_row - 2 - r_in,
                                                )
                                            )
                                            if (
                                                n_sp_movable != 0
                                                and np.sum(
                                                    current_state[
                                                        start_row : end_row + 1, :
                                                    ]
                                                )
                                                < n_targets
                                            ):  # check if there are atoms that can be moved, and if so move them
                                                current_state, _ = move_atoms(
                                                    current_state, space_moves
                                                )
                                                move_set.append(space_moves)
                                                below_moves, n_movable = (
                                                    get_all_moves_btwn_rows(
                                                        current_state,
                                                        stuck_row,
                                                        stuck_row - 1,
                                                    )
                                                )
                                        rows_in += 1
                                    if (
                                        np.sum(
                                            current_state[start_row : end_row + 1, :]
                                        )
                                        < n_targets
                                    ):
                                        current_state, _ = move_atoms(
                                            current_state, below_moves
                                        )
                            if len(below_moves) > 0:
                                move_set.append(below_moves)
                        if n_movable > 0:
                            n_movable_below = n_movable
                        row_offset += 1
                        if len(move_set) > 0:
                            round_moves.extend(move_set)
                    except IndexError:
                        row_offset += 1
                        break
            moves.extend(round_moves)
        if np.sum(current_state[start_row : end_row + 1, :]) >= n_targets:
            success_flag = True
        return moves, col_compact, success_flag

    else:
        col_compact = True

        current_state = copy.deepcopy(init_config)
        while np.sum(current_state[:, start_col : end_col + 1]) < n_targets:
            round_moves = []

            # MOVING FROM ABOVE
            n_movable_above = 0
            col_offset = 0
            while n_movable_above == 0:
                try:
                    move_set = []
                    for off in range(col_offset + 1)[::-1]:
                        above_moves, n_movable = get_all_moves_btwn_cols(
                            current_state, start_col - 1 - off, start_col - off
                        )
                        if (
                            n_movable != 0
                            and np.sum(current_state[:, start_col : end_col + 1])
                            < n_targets
                        ):  # check if there are atoms that can be moved, and if so move them
                            current_state, _ = move_atoms(current_state, above_moves)
                        else:  # if no atoms can be moved, figure out why
                            n_in_from_col = np.sum(
                                current_state[:, start_col - 1 - off]
                            )
                            if (
                                n_in_from_col > 0
                            ):  # if there are no spots for new atoms to come, make space by pushing atoms farther inside
                                cols_in = 0
                                while n_movable == 0:
                                    stuck_col = start_col - off
                                    for c_in in range(cols_in + 1)[::-1]:
                                        space_moves, n_sp_movable = (
                                            get_all_moves_btwn_cols(
                                                current_state,
                                                stuck_col + 1 + c_in,
                                                stuck_col + 2 + c_in,
                                            )
                                        )
                                        if (
                                            n_sp_movable != 0
                                            and np.sum(
                                                current_state[
                                                    :, start_col : end_col + 1
                                                ]
                                            )
                                            < n_targets
                                        ):  # check if there are atoms that can be moved, and if so move them
                                            current_state, _ = move_atoms(
                                                current_state, space_moves
                                            )
                                            move_set.append(space_moves)
                                            above_moves, n_movable = (
                                                get_all_moves_btwn_cols(
                                                    current_state,
                                                    stuck_col,
                                                    stuck_col + 1,
                                                )
                                            )
                                    cols_in += 1
                                if (
                                    np.sum(current_state[:, start_col : end_col + 1])
                                    < n_targets
                                ):
                                    current_state, _ = move_atoms(
                                        current_state, above_moves
                                    )
                        if len(above_moves) > 0:
                            move_set.append(above_moves)
                    if n_movable > 0:
                        n_movable_above = n_movable
                    col_offset += 1
                    if len(move_set) > 0:
                        round_moves.extend(move_set)
                except IndexError:
                    col_offset += 1
                    break

            # MOVING FROM BELOW
            if np.sum(current_state[:, start_row : end_row + 1]) < n_targets:

                # get atoms from below
                n_movable_below = 0
                col_offset = 0
                while n_movable_below == 0:
                    try:
                        move_set = []
                        for off in range(col_offset + 1)[::-1]:
                            below_moves, n_movable = get_all_moves_btwn_cols(
                                current_state, end_col + 1 + off, end_col + off
                            )
                            if (
                                n_movable != 0
                                and np.sum(current_state[:, start_col : end_col + 1])
                                < n_targets
                            ):  # check if there are atoms that can be moved, and if so move them
                                current_state, _ = move_atoms(
                                    current_state, below_moves
                                )
                            else:  # if no atoms can be moved, figure out why
                                n_in_from_col = np.sum(
                                    current_state[end_col + 1 + off, :]
                                )
                                if (
                                    n_in_from_col > 0
                                ):  # if there are no spots for new atoms to come, make space by pushing atoms farther inside
                                    cols_in = 0
                                    while n_movable == 0:
                                        stuck_col = end_col + off
                                        for c_in in range(cols_in + 1)[::-1]:
                                            space_moves, n_sp_movable = (
                                                get_all_moves_btwn_cols(
                                                    current_state,
                                                    stuck_col - 1 - c_in,
                                                    stuck_col - 2 - c_in,
                                                )
                                            )
                                            if (
                                                n_sp_movable != 0
                                                and np.sum(
                                                    current_state[
                                                        :, start_col : end_col + 1
                                                    ]
                                                )
                                                < n_targets
                                            ):  # check if there are atoms that can be moved, and if so move them
                                                current_state, _ = move_atoms(
                                                    current_state, space_moves
                                                )
                                                move_set.append(space_moves)
                                                below_moves, n_movable = (
                                                    get_all_moves_btwn_cols(
                                                        current_state,
                                                        stuck_col,
                                                        stuck_col - 1,
                                                    )
                                                )
                                        cols_in += 1
                                    if (
                                        np.sum(
                                            current_state[:, start_col : end_col + 1]
                                        )
                                        < n_targets
                                    ):
                                        current_state, _ = move_atoms(
                                            current_state, below_moves
                                        )
                            if len(below_moves) > 0:
                                move_set.append(below_moves)
                        if n_movable > 0:
                            n_movable_below = n_movable
                        col_offset += 1
                        if len(move_set) > 0:
                            round_moves.extend(move_set)
                    except IndexError:
                        col_offset += 1
                        break
            moves.extend(round_moves)
        if np.sum(current_state[:, start_col : end_col + 1]) >= n_targets:
            success_flag = True
        return moves, col_compact, success_flag


def get_all_moves_btwn_rows(init_config, from_row_ind, to_row_ind):
    from_row = init_config[from_row_ind, :]
    to_row = init_config[to_row_ind, :]

    available_source = np.where(from_row == 1)[0]
    available_spots = np.where(to_row == 0)[0]

    moves = []
    for atom_col in available_source:
        move = None
        if atom_col - 1 in available_spots:
            move = Move(from_row_ind, atom_col, to_row_ind, atom_col - 1)
            available_spots = available_spots[~np.isin(available_spots, atom_col - 1)]
        elif atom_col in available_spots:
            move = Move(from_row_ind, atom_col, to_row_ind, atom_col)
            available_spots = available_spots[~np.isin(available_spots, atom_col)]
        elif atom_col + 1 in available_spots:
            move = Move(from_row_ind, atom_col, to_row_ind, atom_col + 1)
            available_spots = available_spots[~np.isin(available_spots, atom_col + 1)]
        if move is not None:
            moves.append(move)
    n_atoms_movable = len(moves)
    return moves, n_atoms_movable


def get_all_moves_btwn_cols(init_config, from_col_ind, to_col_ind):
    from_col = init_config[:, from_col_ind]
    to_col = init_config[:, to_col_ind, :]

    available_source = np.where(from_col == 1)[0]
    available_spots = np.where(to_col == 0)[0]

    moves = []
    for atom_row in available_source:
        move = None
        if atom_row - 1 in available_spots:
            move = Move(atom_row, from_col_ind, atom_row - 1, to_col_ind)
            available_spots = available_spots[~np.isin(available_spots, atom_row - 1)]
        elif atom_row in available_spots:
            move = Move(atom_row, from_col_ind, atom_row, to_col_ind)
            available_spots = available_spots[~np.isin(available_spots, atom_row)]
        elif atom_row + 1 in available_spots:
            move = Move(atom_row, from_col_ind, atom_row + 1, to_col_ind)
            available_spots = available_spots[~np.isin(available_spots, atom_row + 1)]
        if move is not None:
            moves.append(move)
    n_atoms_movable = len(moves)
    return moves, n_atoms_movable


def move_across_rows(
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
        # low_ind_roi = m
        # high_ind_roi = j + 1
        low_ind_source = i
        high_ind_source = m
    elif dir == -1:
        start_row = m
        end_row = m - 1
        # low_ind_roi = i
        # high_ind_roi = m
        low_ind_source = m
        high_ind_source = j + 1

    ## sanity check to make sure we have sufficient atoms
    n_atoms_in_source = np.sum(current_state[low_ind_source:high_ind_source])
    # n_atoms_in_roi = np.sum(current_state[low_ind_roi:high_ind_roi]) # linting error - not used
    if n_atoms_in_source < n_to_move:
        raise Exception(
            f"Insufficient atoms. Only {n_atoms_in_source} in the source region."
        )

    ## continue looping until we move sufficient atoms.
    while n_left_to_move != 0:
        n_movable_dir = 0
        row_offset = 0

        ## we loop until we are able to move atoms
        while n_movable_dir == 0:
            try:
                move_set = []
                for off in range(row_offset + 1)[::-1]:
                    across_move = 1
                    from_row = start_row + (off * dir)
                    to_row = end_row + (off * dir)
                    if i > from_row or i > to_row or j < from_row or j < to_row:
                        raise IndexError
                    above_moves, n_movable = get_all_moves_btwn_rows(
                        current_state, from_row, to_row
                    )

                    if (
                        n_movable != 0 and n_left_to_move != 0
                    ):  # check if there are atoms that can be moved, and if so move them
                        if off == 0:
                            moves_to_run = above_moves[:n_left_to_move]
                            current_state, _ = move_atoms(current_state, moves_to_run)
                        else:
                            current_state, _ = move_atoms(current_state, above_moves)
                    else:  # if atoms CANNOT be moved
                        across_move = 0
                        n_in_from_row = np.sum(current_state[from_row, :])
                        # print(f'Stuck: n_atoms in from row are {n_in_from_row}')
                        ## Scenario 1: there are atoms to move, but no place to put them in the new row, so we have to clear room in ROI
                        if n_in_from_row > 0:
                            rows_into_ROI = 0
                            while n_movable == 0:
                                stuck_row = start_row + dir * off
                                for r_in in range(rows_into_ROI + 1)[::-1]:
                                    from_row = stuck_row + (1 + r_in) * dir
                                    to_row = stuck_row + (2 + r_in) * dir
                                    if (
                                        i > from_row
                                        or i > to_row
                                        or j < from_row
                                        or j < to_row
                                    ):
                                        raise IndexError
                                    space_moves, n_sp_movable = get_all_moves_btwn_rows(
                                        current_state, from_row, to_row
                                    )
                                    if (
                                        n_sp_movable != 0 and n_left_to_move != 0
                                    ):  # check if there are atoms that can be moved, and if so move them
                                        current_state, _ = move_atoms(
                                            current_state, space_moves
                                        )
                                        move_set.append(space_moves)
                                        n_movable = n_sp_movable
                                rows_into_ROI += 1
                        ## Scenario 2: there are no atoms to move, so we have to take atoms from farther inside the source region
                        elif n_in_from_row == 0:
                            rows_into_source = 0
                            while n_movable == 0:
                                stuck_row = start_row + dir * off
                                for r_in in range(rows_into_source + 1)[::-1]:
                                    from_row = stuck_row - (1 + r_in) * dir
                                    to_row = stuck_row - (r_in) * dir
                                    if (
                                        i > from_row
                                        or i > to_row
                                        or j < from_row
                                        or j < to_row
                                    ):
                                        raise IndexError
                                    space_moves, n_sp_movable = get_all_moves_btwn_rows(
                                        current_state, from_row, to_row
                                    )
                                    if (
                                        n_sp_movable != 0 and n_left_to_move != 0
                                    ):  # check if there are atoms that can be moved, and if so move them
                                        current_state, _ = move_atoms(
                                            current_state, space_moves
                                        )
                                        move_set.append(space_moves)
                                        n_movable = n_sp_movable
                                rows_into_source += 1
                    if len(above_moves) > 0:
                        if off == 0 and n_left_to_move != 0 and across_move:
                            moves_to_run = above_moves[:n_left_to_move]
                            move_set.append(moves_to_run)
                            n_left_to_move -= len(moves_to_run)
                        else:
                            move_set.append(above_moves)

                if len(move_set) > 0:
                    round_moves.extend(move_set)
                if n_movable > 0:
                    n_movable_dir = n_movable  # is the next line equivalent
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
    difference = j - i + 1
    m = i + (difference // 2)
    next_list = []
    if i != j and i < j:
        next_list.append((i, m - 1))
        next_list.append((m, j))
    return next_list
