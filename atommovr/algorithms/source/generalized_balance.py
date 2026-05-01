##Algorithm for the Balance and Compact Algorithm
import copy
import numpy as np
from collections import deque

from atommovr.utils.move_utils import Move, move_atoms

from atommovr.algorithms.source.ejection import ejection


def generalized_balance(
    init_config,
    target_config,
    do_ejection: bool = False,
    final_size: list[int] | None = None,
):
    generalized_balance_success_flag = False
    matrix = copy.deepcopy(init_config)
    move_list = []

    # Counts moves components in the Hungarian algorithm
    # balance_moves_term = 0 # linting error - unused
    # ejection_moves_term = 0

    if final_size is None:
        final_size = [0, len(matrix[0]) - 1, 0, len(matrix) - 1]
    elif len(final_size) == 0:
        final_size = [0, len(matrix[0]) - 1, 0, len(matrix) - 1]

    row_min = final_size[0]
    row_max = final_size[1]
    col_min = final_size[2]
    col_max = final_size[3]

    balance_config, move_list = row_balance(
        matrix, target_config, row_min, row_max, col_min, col_max, move_list, 0
    )
    # balance_moves_term = len(move_list)
    final_config = copy.deepcopy(balance_config)

    if do_ejection:
        eject_moves, final_config = ejection(balance_config, target_config, final_size)
        move_list.extend(eject_moves)
        # ejection_moves_term = len(eject_moves)
        # Check if the configuration is the same as the target configuration
        if np.array_equal(final_config, target_config):
            generalized_balance_success_flag = True
    else:
        # ejection_moves_term = 0
        # Check if the configuration (inside range of target) the same as the target configuration
        effective_config = np.multiply(final_config, target_config)
        if np.array_equal(effective_config, target_config):
            generalized_balance_success_flag = True

    return (
        final_config,
        move_list,
        generalized_balance_success_flag,
    )  # , [balance_moves_term, ejection_moves_term]


def row_balance(
    matrix, target_config, row_min, row_max, col_min, col_max, move_list, recursive_flag
):
    # 1. Top_Bottom Lattice Balance
    # Calculate the number of rows in the submatrix
    row_nums = row_max - row_min + 1

    if row_nums == 1 and recursive_flag == 0:
        return col_balance(
            matrix, target_config, row_min, row_max, col_min, col_max, move_list, 1
        )

    # 2. Left_Right Lattice Balance
    # Calculate the middle row index
    middle_row = row_min + (row_nums // 2) - 1
    # middle_col = col_min + (col_nums // 2) - 1

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
        move_list, balance_config = down_move(
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
        move_list, balance_config = up_move(
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
        balance_config, move_list = col_balance(
            balance_config,
            target_config,
            row_min,
            middle_row,
            col_min,
            col_max,
            move_list,
            0,
        )
        balance_config, move_list = col_balance(
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


def col_balance(
    matrix, target_config, row_min, row_max, col_min, col_max, move_list, recursive_flag
):
    # Recursively balance the submatrices
    col_nums = col_max - col_min + 1

    if col_nums == 1 and recursive_flag == 0:
        return row_balance(
            matrix, target_config, row_min, row_max, col_min, col_max, move_list, 1
        )

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

    if recursive_flag == 0:
        balance_config, move_list = row_balance(
            balance_config,
            target_config,
            row_min,
            row_max,
            col_min,
            middle_col,
            move_list,
            0,
        )
        balance_config, move_list = row_balance(
            balance_config,
            target_config,
            row_min,
            row_max,
            middle_col + 1,
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
            and target_row + stuff < len(matrix)
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

            ## Slide up move to achieve balance
            # moves_in_scan = []
            # for col in range(col_min, col_max + 1):
            #     if matrix[source_row, col] == 1 and matrix[source_row + 1, col - 1] == 0 and matrix[target_row, col] == 0 and excess_atoms > 0:
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

    return move_list, matrix


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

        # if source_row > row_max and needed_atoms > 0:
        ## Slide up move to achieve balance
        # print(f"Up move, Lattice1:{row_min}, {middle_row}, {col_min}, {col_max}")
        # print(f"Up move, Lattice2:{middle_row+1}, {row_max}, {col_min}, {col_max}")
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

    return move_list, matrix


def right_move(
    matrix,
    target_config,
    excess_atoms,
    row_min,
    row_max,
    col_min,
    middle_col,
    col_max,
    move_list,
):
    # Initialize the move bound
    source_col = middle_col
    target_col = middle_col + 1
    normalize_col = 0
    balance_col_count = target_col - source_col
    stuff = 0
    brute_force_start = []
    brute_force_end = []

    # while excess_atoms >  0:
    while col_max >= source_col >= col_min and excess_atoms > 0:
        moves_in_scan = []

        for row in range(row_min, row_max + 1):
            if (
                matrix[row, source_col + normalize_col] == 1
                and matrix[row, source_col + normalize_col + 1] == 0
                and matrix[row, target_col] == 0
                and excess_atoms > 0
            ):
                move = Move(
                    row, source_col + normalize_col, row, source_col + normalize_col + 1
                )
                moves_in_scan.append(move)

                if balance_col_count == 1:
                    excess_atoms -= 1

        if len(moves_in_scan) > 0:
            move_list.append(moves_in_scan)
            matrix, _ = move_atoms(matrix, moves_in_scan)

        # TODO: Solve the stuff case later
        if (
            sum(matrix[row_min : row_max + 1, target_col])
            == len(matrix[row_min : row_max + 1, target_col])
            and target_col + stuff < len(matrix)
            and excess_atoms > 0
        ):
            for shift in range(stuff, -1, -1):
                moves_in_scan = []
                # Skip shifts whose one-step-right destination would leave the grid.
                # TODO: Audit right-edge termination logic; this guard prevents an IndexError
                # but does not prove the surrounding right_move boundary logic is correct.
                if target_col + shift + 1 >= len(matrix[0]):
                    continue
                for row in range(row_min, row_max + 1):
                    if (
                        matrix[row, target_col + shift] == 1
                        and matrix[row, target_col + shift + 1] == 0
                    ):
                        move = Move(
                            row, target_col + shift, row, target_col + shift + 1
                        )
                        moves_in_scan.append(move)

                if len(moves_in_scan) > 0:
                    move_list.append(moves_in_scan)
                    matrix, _ = move_atoms(matrix, moves_in_scan)

            stuff += 1
            source_col = middle_col
            balance_col_count = target_col - source_col
            normalize_col = 0
        else:
            balance_col_count -= 1
            normalize_col += 1

        if balance_col_count == 0:
            source_col -= 1
            balance_col_count = target_col - source_col
            normalize_col = 0

            ## Slide up move to achieve balance
            # moves_in_scan = []
            # for col in range(col_min, col_max + 1):
            #     if matrix[source_row, col] == 1 and matrix[source_row + 1, col - 1] == 0 and matrix[target_row, col] == 0 and excess_atoms > 0:
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

    return move_list, matrix


def left_move(
    matrix,
    target_config,
    excess_atoms,
    row_min,
    row_max,
    col_min,
    middle_col,
    col_max,
    move_list,
):
    # Initialize the move bound
    source_col = middle_col + 1
    target_col = middle_col
    normalize_col = 0
    balance_col_count = source_col - target_col
    stuff = 0
    brute_force_start = []
    brute_force_end = []

    # while needed_atoms > 0:
    while col_max >= source_col >= col_min and excess_atoms > 0:
        moves_in_scan = []

        # Iterate through the rows in the submatrix
        for row in range(row_min, row_max + 1):
            # Check if the atom is in the source column and the target column is empty
            if (
                matrix[row, source_col + normalize_col] == 1
                and matrix[row, source_col + normalize_col - 1] == 0
                and matrix[row, target_col] == 0
                and excess_atoms > 0
            ):
                move = Move(
                    row, source_col + normalize_col, row, source_col + normalize_col - 1
                )
                # print(f'{move.from_row}, {move.from_col} -> {move.to_row}, {move.to_col}')
                moves_in_scan.append(move)

                # Decrement the number of excess atoms
                if balance_col_count == 1:
                    excess_atoms -= 1

        # If there are moves in the scan, add them to the move list and update the matrix
        if len(moves_in_scan) > 0:
            move_list.append(moves_in_scan)
            matrix, _ = move_atoms(matrix, moves_in_scan)

        # Check if the target column is full and there are still excess atoms
        if (
            sum(matrix[row_min : row_max + 1, target_col])
            == len(matrix[row_min : row_max + 1, target_col])
            and target_col > 0
            and excess_atoms > 0
        ):
            for shift in range(stuff, -1, -1):
                moves_in_scan = []
                for row in range(row_min, row_max + 1):
                    if (
                        matrix[row, target_col - shift] == 1
                        and matrix[row, target_col - shift - 1] == 0
                    ):
                        move = Move(
                            row, target_col - shift, row, target_col - shift - 1
                        )
                        # print(f'{move.from_row}, {move.from_col} -> {move.to_row}, {move.to_col}')
                        moves_in_scan.append(move)

                if len(moves_in_scan) > 0:
                    move_list.append(moves_in_scan)
                    matrix, _ = move_atoms(matrix, moves_in_scan)

            stuff += 1
            source_col = middle_col + 1
            balance_col_count = source_col - target_col
            normalize_col = 0
        else:
            balance_col_count -= 1
            normalize_col -= 1

        if balance_col_count == 0:
            source_col += 1
            balance_col_count = source_col - target_col
            normalize_col = 0
        # if source_col > col_max and excess_atoms > 0:
        ## Slide up move to achieve balance
        # fig_atom_arrays(matrix)
        # print(f"Left move, Lattice1:{row_min}, {row_max}, {col_min}, {middle_col}")
        # print(f"Left move, Lattice2:{row_min}, {row_max}, {middle_col+1}, {col_max}")

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

    return move_list, matrix


##Find possible path between start and end position
def bfs_move_atom(grid, start, end):
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
            return path

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

    # [Path between start and obstacle] + [Path between obstacle and end]
    return path + [obstacle], bfs_move_atom(grid, obstacle, end)


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
