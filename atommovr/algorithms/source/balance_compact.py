import copy
import numpy as np

##Algorithm for the Balance and Compact Algorithm
from atommovr.utils.move_utils import Move, move_atoms

from atommovr.algorithms.source.ejection import ejection
from atommovr.algorithms.source.generalized_balance import (
    right_move,
    left_move,
    flatten_tuple,
    bfs_move_atom,
)


def balance_and_compact(
    init_config: np.ndarray,
    target_config: np.ndarray,
    do_ejection: bool = False,
    final_size: list[int] | None = None,
):
    if len(np.shape(init_config)) > 2 and np.shape(init_config)[2] == 1:
        matrix = np.array(copy.deepcopy(init_config[:, :, 0]))
    elif len(np.shape(init_config)) == 2:
        matrix = np.array(copy.deepcopy(init_config))
    else:
        raise ValueError(
            f"Atom array has shape {np.shape(init_config)}, which is not correct for single species. Did you meant to use a dual species algorithm?"
        )
    success_flag = False

    row_min = 0
    row_max = len(matrix) - 1

    # 1. Balance left part algorithm and right part algorithm
    move_list = []
    col_min = 0
    col_max = len(matrix[0]) - 1
    col_nums = col_max - col_min + 1
    middle_col = col_min + (col_nums // 2) - 1

    # Counts moves components in the balance and compact algorithm
    # balance_moves_term = 0
    # compact_moves_term = 0
    # ejection_moves_term = 0

    pre_balance_config, move_list = pre_balance(
        matrix, target_config, row_min, row_max, col_min, col_max, move_list
    )
    balance_left_config, move_list = balance_bc(
        pre_balance_config,
        target_config,
        row_min,
        row_max,
        col_min,
        middle_col,
        move_list,
        0,
    )
    balance_config, move_list = balance_bc(
        balance_left_config,
        target_config,
        row_min,
        row_max,
        middle_col + 1,
        col_max,
        move_list,
        0,
    )
    # balance_moves_term = len(move_list)

    # 2. Compact part alogrithm
    compact_config, move_list = compact_left(
        balance_config, target_config, middle_col, move_list
    )
    compact_config, move_list = compact_right(
        compact_config, target_config, middle_col, move_list
    )
    # compact_moves_term = len(move_list) - balance_moves_term # linting error - unused

    final_config = copy.deepcopy(compact_config)

    # 3 Eject to certain geoemetry
    if do_ejection:
        eject_moves, final_config = ejection(
            compact_config, target_config, [0, len(matrix) - 1, 0, len(matrix[0]) - 1]
        )
        move_list.extend(eject_moves)
        # ejection_moves_term = len(eject_moves)
        # 3.1 Check if the configuration is the same as the target configuration
        if np.array_equal(final_config, target_config):
            success_flag = True
    else:
        # ejection_moves_term = 0 # linting error - unused
        # 3.2 Check if the configuration (inside range of target) the same as the target configuration
        effective_config = np.multiply(final_config, target_config)
        if np.array_equal(effective_config, target_config):
            success_flag = True

    return (
        final_config,
        move_list,
        success_flag,
    )  # , [balance_moves_term, compact_moves_term, ejection_moves_term]


def pre_balance(matrix, target_config, row_min, row_max, col_min, col_max, move_list):
    # Recursively balance the submatrices
    col_nums = col_max - col_min + 1

    middle_col = col_min + (col_nums // 2) - 1
    n_req = int(np.sum(target_config[row_min : row_max + 1, col_min : middle_col + 1]))
    n_req_c = int(
        np.sum(target_config[row_min : row_max + 1, middle_col + 1 : col_max + 1])
    )
    # print(f"n_req: {n_req}")
    N_S_i_m = int(np.sum(matrix[row_min : row_max + 1, col_min : middle_col + 1]))
    N_S_i_m_c = int(np.sum(matrix[row_min : row_max + 1, middle_col + 1 : col_max + 1]))
    # print(f"Atoms:{N_S_i_m}")

    if N_S_i_m > n_req and N_S_i_m_c < n_req_c:
        # Shift excess 1s down from S[i:m+1] to S[m+1:j+1]
        excess_atoms = min(N_S_i_m - n_req, n_req_c - N_S_i_m_c)
        move_list, balance_config = right_move(
            matrix,
            target_config,
            excess_atoms,
            row_min,
            row_max,
            col_min,
            middle_col,
            col_max,
            move_list,
        )
        # return matrix, move_list

    elif N_S_i_m < n_req and N_S_i_m_c > n_req_c:
        # Shift required 1s up from S[m+1:j+1] to S[i:m+1]
        needed_atoms = min(n_req - N_S_i_m, N_S_i_m_c - n_req_c)
        move_list, balance_config = left_move(
            matrix,
            target_config,
            needed_atoms,
            row_min,
            row_max,
            col_min,
            middle_col,
            col_max,
            move_list,
        )
        # return matrix, move_list
    else:
        balance_config = matrix

    return balance_config, move_list


def balance_bc(
    matrix, target_config, row_min, row_max, col_min, col_max, move_list, recursive_flag
):
    # 1. Top_Bottom Lattice Balance
    # Calculate the number of rows in the submatrix
    row_nums = row_max - row_min + 1

    if row_nums == 1 and recursive_flag == 0:
        return matrix, move_list

    # 2. Left_Right Lattice Balance
    # Calculate the middle row index
    middle_row = row_min + (row_nums // 2) - 1

    # Calculate the required number of 1s in the submatrix S[i:m+1]
    n_req = int(np.sum(target_config[row_min : middle_row + 1, col_min : col_max + 1]))
    n_req_c = int(
        np.sum(target_config[middle_row + 1 : row_max + 1, col_min : col_max + 1])
    )
    # print(f"n_req: {n_req}")

    # Calculate the actual number of 1s in the submatrix S[i:m+1]
    N_S_i_m = int(np.sum(matrix[row_min : middle_row + 1, col_min : col_max + 1]))
    N_S_i_m_c = int(np.sum(matrix[middle_row + 1 : row_max + 1, col_min : col_max + 1]))
    # print(f"Atoms:{N_S_i_m}")

    if N_S_i_m > n_req and N_S_i_m_c < n_req_c:
        # Shift excess 1s down from S[i:m+1] to S[m+1:j+1]
        excess_atoms = min(N_S_i_m - n_req, n_req_c - N_S_i_m_c)
        balance_config, move_list = down_move(
            matrix,
            target_config,
            excess_atoms,
            row_min,
            middle_row,
            row_max,
            col_min,
            col_max,
            move_list,
        )
        # return matrix, move_list

    elif N_S_i_m < n_req and N_S_i_m_c > n_req_c:
        # Shift required 1s up from S[m+1:j+1] to S[i:m+1]
        needed_atoms = min(n_req - N_S_i_m, N_S_i_m_c - n_req_c)
        balance_config, move_list = up_move(
            matrix,
            target_config,
            needed_atoms,
            row_min,
            middle_row,
            row_max,
            col_min,
            col_max,
            move_list,
        )
        # return matrix, move_list
    else:
        balance_config = matrix

    if recursive_flag == 0:
        balance_config, move_list = balance_bc(
            balance_config,
            target_config,
            row_min,
            middle_row,
            col_min,
            col_max,
            move_list,
            0,
        )
        balance_config, move_list = balance_bc(
            balance_config,
            target_config,
            middle_row + 1,
            row_max,
            col_min,
            col_max,
            move_list,
            0,
        )

    return balance_config, move_list


def down_move(
    matrix,
    target_config,
    excess_atoms,
    row_min,
    middle_row,
    row_max,
    col_min,
    col_max,
    move_list,
):
    # Initialize the move bound
    source_row = middle_row
    target_row = middle_row + 1
    normalize_row = 0
    balance_row_count = target_row - source_row
    stuff = 0
    brute_force_start = []
    brute_force_end = []

    # while excess_atoms >  0:
    while row_max >= source_row >= row_min and excess_atoms > 0:
        moves_in_scan = []

        for col in range(col_min, col_max + 1):
            if (
                matrix[source_row + normalize_row, col] == 1
                and matrix[source_row + normalize_row + 1, col] == 0
                and matrix[target_row, col] == 0
                and excess_atoms > 0
            ):
                move = Move(
                    source_row + normalize_row, col, source_row + normalize_row + 1, col
                )
                # print(f'{move.from_row}, {move.from_col} -> {move.to_row}, {move.to_col}')
                moves_in_scan.append(move)

                if balance_row_count == 1:
                    excess_atoms -= 1

        if len(moves_in_scan) > 0:
            move_list.append(moves_in_scan)
            matrix, _ = move_atoms(matrix, moves_in_scan)

        # TODO: Solve the stuff case later
        if (
            sum(matrix[target_row, col_min : col_max + 1])
            == len(matrix[target_row, col_min : col_max + 1])
            and target_row + stuff + 1 < len(matrix)
            and excess_atoms > 0
        ):
            for shift in range(stuff, -1, -1):
                moves_in_scan = []
                for col in range(col_min, col_max + 1):
                    if (
                        matrix[target_row + shift, col] == 1
                        and matrix[target_row + shift + 1, col] == 0
                    ):
                        move = Move(
                            target_row + shift, col, target_row + shift + 1, col
                        )
                        # print(f'{move.from_row}, {move.from_col} -> {move.to_row}, {move.to_col}')
                        moves_in_scan.append(move)

                if len(moves_in_scan) > 0:
                    move_list.append(moves_in_scan)
                    matrix, _ = move_atoms(matrix, moves_in_scan)

            stuff += 1
            source_row = middle_row
            balance_row_count = target_row - source_row
            normalize_row = 0
        else:
            balance_row_count -= 1
            normalize_row += 1

        if balance_row_count == 0:
            source_row -= 1
            balance_row_count = target_row - source_row
            normalize_row = 0

    if excess_atoms > 0:
        # Find unbalance atoms
        for col_ind in range(col_min, col_max + 1):
            for row_ind in range(row_min, row_max + 1):
                if (
                    matrix[row_ind][col_ind] == 1
                    and target_config[row_ind][col_ind] == 0
                ):
                    brute_force_start.append((row_ind, col_ind))
                if (
                    matrix[row_ind][col_ind] == 0
                    and target_config[row_ind][col_ind] == 1
                ):
                    brute_force_end.append((row_ind, col_ind))

        for ind in range(len(brute_force_start)):
            if ind < len(brute_force_end):
                path = flatten_tuple(
                    bfs_move_atom(matrix, brute_force_start[ind], brute_force_end[ind])
                )
                path = path[::-1]

            # Iterate all path segments (((a1,b1), (a2,b2)), ((c1,c2), (d1,d2)))
            for item in path:
                current_pos = item[0]
                for next_pos in item:
                    if np.array_equal(current_pos, next_pos):
                        pass
                    else:
                        if (
                            matrix[current_pos[0]][current_pos[1]] == 1
                            and matrix[next_pos[0]][next_pos[1]] == 0
                        ):
                            matrix[current_pos[0]][current_pos[1]] = 0
                            matrix[next_pos[0]][next_pos[1]] = 1
                            move_list.append(
                                [
                                    Move(
                                        current_pos[0],
                                        current_pos[1],
                                        next_pos[0],
                                        next_pos[1],
                                    )
                                ]
                            )
                        current_pos = next_pos

    return matrix, move_list


def up_move(
    matrix,
    target_config,
    needed_atoms,
    row_min,
    middle_row,
    row_max,
    col_min,
    col_max,
    move_list,
):
    # Initialize the move bound
    source_row = middle_row + 1
    target_row = middle_row
    normalize_row = 0
    balance_row_count = source_row - target_row
    stuff = 0
    brute_force_start = []
    brute_force_end = []

    # while needed_atoms > 0:
    while row_max >= source_row >= row_min and needed_atoms > 0:
        moves_in_scan = []
        for col in range(col_min, col_max + 1):
            if (
                matrix[source_row + normalize_row, col] == 1
                and matrix[source_row + normalize_row - 1, col] == 0
                and matrix[target_row, col] == 0
                and needed_atoms > 0
            ):
                move = Move(
                    source_row + normalize_row, col, source_row + normalize_row - 1, col
                )
                # print(f'{move.from_row}, {move.from_col} -> {move.to_row}, {move.to_col}')
                moves_in_scan.append(move)

                if balance_row_count == 1:
                    needed_atoms -= 1

        if len(moves_in_scan) > 0:
            move_list.append(moves_in_scan)
            matrix, _ = move_atoms(matrix, moves_in_scan)

        if (
            sum(matrix[target_row, col_min : col_max + 1])
            == len(matrix[target_row, col_min : col_max + 1])
            and target_row > 0
            and needed_atoms > 0
        ):
            for shift in range(stuff, -1, -1):
                moves_in_scan = []
                for col in range(col_min, col_max + 1):
                    if (
                        matrix[target_row - shift, col] == 1
                        and matrix[target_row - shift - 1, col] == 0
                    ):
                        move = Move(
                            target_row - shift, col, target_row - shift - 1, col
                        )
                        # print(f'{move.from_row}, {move.from_col} -> {move.to_row}, {move.to_col}')
                        moves_in_scan.append(move)

                if len(moves_in_scan) > 0:
                    move_list.append(moves_in_scan)
                    matrix, _ = move_atoms(matrix, moves_in_scan)

            stuff += 1
            source_row = middle_row + 1
            balance_row_count = source_row - target_row
            normalize_row = 0
        else:
            balance_row_count -= 1
            normalize_row -= 1

        if balance_row_count == 0:
            source_row += 1
            balance_row_count = source_row - target_row
            normalize_row = 0

    if needed_atoms > 0:
        # Find unbalance atoms
        for col_ind in range(col_min, col_max + 1):
            for row_ind in range(row_min, row_max + 1):
                if (
                    matrix[row_ind][col_ind] == 1
                    and target_config[row_ind][col_ind] == 0
                ):
                    brute_force_start.append((row_ind, col_ind))
                if (
                    matrix[row_ind][col_ind] == 0
                    and target_config[row_ind][col_ind] == 1
                ):
                    brute_force_end.append((row_ind, col_ind))
        for ind in range(len(brute_force_start)):
            if ind < len(brute_force_end):
                path = flatten_tuple(
                    bfs_move_atom(matrix, brute_force_start[ind], brute_force_end[ind])
                )
                path = path[::-1]

            # Iterate all path segments (((a1,b1), (a2,b2)), ((c1,c2), (d1,d2)))
            for item in path:
                current_pos = item[0]
                for next_pos in item:
                    if np.array_equal(current_pos, next_pos):
                        pass
                    else:
                        if (
                            matrix[current_pos[0]][current_pos[1]] == 1
                            and matrix[next_pos[0]][next_pos[1]] == 0
                        ):
                            matrix[current_pos[0]][current_pos[1]] = 0
                            matrix[next_pos[0]][next_pos[1]] = 1
                            move_list.append(
                                [
                                    Move(
                                        current_pos[0],
                                        current_pos[1],
                                        next_pos[0],
                                        next_pos[1],
                                    )
                                ]
                            )
                        current_pos = next_pos

    return matrix, move_list


def compact_left(init_matrix, target_config, middle_col, move_list, max_cycles=30):
    matrix = copy.deepcopy(init_matrix)

    left_rearrangement_cycle = 0
    success_flag = False

    # Execute Parallel Sorting until Defect-free or until max number of cycles
    # Execute Parallel Sorting until Defect-free or until max number of cycles
    while not success_flag:  # success_flag == False: # linting error
        if left_rearrangement_cycle == max_cycles:
            return matrix, move_list

        # Moving atoms rightward, iterating col by col
        for col_ind in range(middle_col - 1, -1, -1):
            right_moves = []
            col = matrix[:, col_ind]
            # check if the atom should be moved right, and if so note down the move
            for row_ind, spot in enumerate(col):
                num_target_spots = np.sum(
                    target_config[row_ind, col_ind : middle_col + 1]
                )
                num_atoms = np.sum(matrix[row_ind, col_ind : middle_col + 1])
                if (
                    spot == 0
                    or num_target_spots < num_atoms
                    or matrix[row_ind, col_ind + 1] == 1
                ):
                    pass
                else:
                    right_moves.append(Move(row_ind, col_ind, row_ind, col_ind + 1))

            if len(right_moves) > 0:
                matrix, _ = move_atoms(matrix, right_moves)
                move_list.append(right_moves)

        left_rearrangement_cycle += 1

    return matrix, move_list


def compact_right(init_matrix, target_config, middle_col, move_list, max_cycles=30):
    matrix = copy.deepcopy(init_matrix)

    right_rearrangement_cycle = 0
    success_flag = False

    while not success_flag:  # success_flag == False: # linting error
        if (
            right_rearrangement_cycle == max_cycles or success_flag
        ):  # success_flag == True: #linting error
            return matrix, move_list

        # Moving atoms leftward, iterating col by col
        for col_ind in range(middle_col + 2, len(matrix[0])):
            left_moves = []
            col = matrix[:, col_ind]
            # check if the atom should be moved left, and if so note down the move
            for row_ind, spot in enumerate(col):
                num_target_spots = np.sum(
                    target_config[row_ind, middle_col + 1 : col_ind]
                )
                num_atoms = np.sum(matrix[row_ind, middle_col + 1 : col_ind])
                if (
                    spot == 0
                    or num_target_spots < num_atoms
                    or matrix[row_ind, col_ind - 1] == 1
                ):
                    pass
                else:
                    # left_moves.append(((i,j),(i-1,j)))
                    left_moves.append(Move(row_ind, col_ind, row_ind, col_ind - 1))
            if len(left_moves) > 0:
                matrix, _ = move_atoms(matrix, left_moves)
                move_list.append(left_moves)

        target_config_check = np.multiply(target_config, matrix)

        if np.array_equal(target_config_check, target_config):
            success_flag = True

        right_rearrangement_cycle += 1

    # right_rearrangement_cycle = 15
    # success_flag == False

    return matrix, move_list
