# Code to extract lower bounds

import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

from atommovr.algorithms.source.PPSU_weight_matching import bttl_threshold


def calculate_LB(
    atom_arrays: np.ndarray, target_config: np.ndarray, n_species: int = 1
):
    mat_copy = copy.deepcopy(atom_arrays)
    if n_species == 1:
        mat_copy.reshape([len(mat_copy), len(mat_copy[0]), 1])

    LBs = []
    for species_ind in range(np.shape(mat_copy)[2]):
        matrix = mat_copy[:, :, species_ind]

        # Define target positions for the center square in a matrix.
        current_positions, target_positions = define_current_and_target(
            matrix, target_config
        )

        # Generate the cost matrix using the current atom positions and the target positions
        cost_matrix = generate_cost_matrix(current_positions, target_positions)
        sq_matrix = make_cost_matrix_square(cost_matrix)

        # Solve the linear BOTTLENECK assignment problem to find the lower bound rearrangement time
        LB = get_Zstar_lower_bound(sq_matrix)
        LBs.append(LB)

    return np.max(LBs), mat_copy


def make_cost_matrix_square(cost_matrix):
    shape = np.shape(cost_matrix)
    # making matrix square if it is rectangular
    if shape[0] != shape[1]:
        new_matrix = np.zeros([np.max(shape), np.max(shape)])
        new_matrix[: shape[0], : shape[1]] = cost_matrix
    else:
        new_matrix = copy.deepcopy(cost_matrix)
    return new_matrix


def get_LBAP_pairing(
    atom_arrays: np.ndarray, target_config: np.ndarray, n_species=1, metric="euclidean"
):
    mat_copy = copy.deepcopy(atom_arrays)
    targ_copy = copy.deepcopy(target_config)
    if n_species == 1:
        mat_copy.reshape([len(mat_copy), len(mat_copy[0]), 1])
        targ_copy.reshape([len(targ_copy), len(targ_copy[0]), 1])

    pairs = []
    for species_ind in range(np.shape(mat_copy)[2]):
        matrix = mat_copy[:, :, species_ind]
        target = targ_copy[:, :, species_ind]

        # Define target positions for the center square in a matrix.
        current_positions, target_positions = define_current_and_target(matrix, target)

        # Generate the cost matrix using the current atom positions and the target positions
        cost_matrix = generate_cost_matrix(
            current_positions, target_positions, metric=metric
        )
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
        for row_ind in range(len(sq_cost)):
            col_ind = col_inds[row_ind]
            current_pos = current_positions[row_ind]
            try:
                target_pos = target_positions[col_ind]
                pairs.append([current_pos, target_pos])
            except IndexError:
                pass

    return pairs


def calculate_Zstar_better(
    atom_arrays: np.ndarray, target_config: np.ndarray, n_species=1, metric="euclidean"
):
    mat_copy = copy.deepcopy(atom_arrays)
    targ_copy = copy.deepcopy(target_config)
    if n_species == 1:
        mat_copy.reshape([len(mat_copy), len(mat_copy[0]), 1])
        targ_copy.reshape([len(targ_copy), len(targ_copy[0]), 1])

    zstars = []
    for species_ind in range(np.shape(mat_copy)[2]):
        matrix = mat_copy[:, :, species_ind]
        target = targ_copy[:, :, species_ind]

        # Define target positions for the center square in a matrix.
        current_positions, target_positions = define_current_and_target_new(
            matrix, target
        )

        # Generate the cost matrix using the current atom positions and the target positions
        cost_matrix = generate_cost_matrix(
            current_positions, target_positions, metric=metric
        )
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
        costs = []
        for row_ind in range(len(sq_cost)):
            col_ind = col_inds[row_ind]
            costs.append(sq_cost[row_ind, col_ind])

        zstars.append(np.max(costs))

    return np.max(zstars)


def calculate_Zstar(atom_arrays: np.ndarray, target_config: np.ndarray, n_species=1):
    mat_copy = copy.deepcopy(atom_arrays)
    if n_species == 1:
        mat_copy.reshape([len(mat_copy), len(mat_copy[0]), 1])

    LBs = []
    zstars = []
    for species_ind in range(np.shape(mat_copy)[2]):
        matrix = mat_copy[:, :, species_ind]

        # Define target positions for the center square in a matrix.
        current_positions, target_positions = define_current_and_target(
            matrix, target_config
        )

        # Generate the cost matrix using the current atom positions and the target positions
        cost_matrix = generate_cost_matrix(current_positions, target_positions)
        sq_cost = make_cost_matrix_square(cost_matrix)

        # Solve the linear BOTTLENECK assignment problem to find the lower bound rearrangement time
        f_matrix, LB, mat_copy, zstar_fail_flag = convert_LBAP_to_LSAP(sq_cost)
        LBs.append(LB)

        # row_ind and col_ind are arrays of indices indicating the optimal assignment
        try:
            row_ind, col_ind = linear_sum_assignment(f_matrix)
        except ValueError:
            row_ind, col_ind = [], []

        if not zstar_fail_flag:
            costs = sq_cost[row_ind, col_ind]
            largest_distance = np.max(costs)
            if LB > largest_distance:
                raise Exception(
                    f"Algorithm error. Lower bound {LB} cannot be larger than Z* {largest_distance}."
                )
        else:
            largest_distance = 0

        zstars.append(largest_distance)

    # # Pair up row_ind and col_ind and sort by col_ind
    # paired_indices = sorted(zip(row_ind, col_ind), key=lambda x: x[1])

    # largest_distance = 0
    # for i, pair in enumerate(paired_indices):
    #     try:
    #         distance = cost_matrix[pair[0], pair[1]]
    #         if distance > largest_distance:
    #             largest_distance = distance
    #     except IndexError:
    #         print(f'Error: index {pair} = {[row_ind[i],col_ind[i]]} tried to access matrix with shape {np.shape(cost_matrix)}')

    return np.max(zstars), np.max(LBs), zstar_fail_flag


def define_current_and_target(matrix, target_config):
    current_positions = [
        (x, y)
        for x in range(len(matrix[0]))
        for y in range(len(matrix))
        if matrix[y][x] == 1
        if target_config[y][x] == 0
    ]  # NKH: this used to be matrix[x,y] and target_config[x,y (the row index and column index were mixed up) but I fixed it
    target_positions = [
        (x, y)
        for x in range(len(matrix[0]))
        for y in range(len(matrix))
        if matrix[y][x] == 0
        if target_config[y][x] == 1
    ]  # NKH: same here
    return current_positions, target_positions


def define_current_and_target_new(matrix, target_config):
    """
    Returns a list of ALL target positions (including ones that have been filled)
    and ALL current positions (including ones that fill target sites)."""
    current_positions = [
        (x, y)
        for x in range(len(matrix[0]))
        for y in range(len(matrix))
        if matrix[y][x] == 1
    ]  # if target_config[y][x] == 0] # NKH: this used to be matrix[x,y] and target_config[x,y (the row index and column index were mixed up) but I fixed it
    target_positions = [
        (x, y)
        for x in range(len(matrix[0]))
        for y in range(len(matrix))
        if target_config[y][x] == 1
    ]  # if matrix[y][x] == 0] # NKH: same here
    return current_positions, target_positions


# Generate a cost matrix for the Hungarian Algorithm.
def generate_cost_matrix(current_positions, target_positions, metric="euclidean"):
    num_atoms = len(current_positions)
    num_targets = len(target_positions)
    cost_matrix = np.zeros((num_atoms, num_targets))
    if num_atoms < num_targets:
        return cost_matrix

    for i, current in enumerate(current_positions):
        for j, target in enumerate(target_positions):
            if metric == "euclidean":
                cost_matrix[i, j] = np.sqrt(
                    (current[0] - target[0]) ** 2 + (current[1] - target[1]) ** 2
                )
            elif metric == "moves":
                dy = np.abs(current[0] - target[0])
                dx = np.abs(current[1] - target[1])
                min_dist = np.min([dy, dx])
                max_dist = np.max([dy, dx])
                cost = min_dist * np.sqrt(2)
                cost += max_dist - min_dist
                cost_matrix[i, j] = cost
    return cost_matrix


# Converting linear bottleneck assignment problem (LBAP) to linear sum assignment (Hungarian) problem (LSAP)
# Algorithm taken from the following paper:
# Kuo, CC., Nicholls, G. A turnpike approach to solving the linear bottleneck assignment problem.
# Int J Adv Manuf Technol 71, 1059-1068 (2014). https://doi.org/10.1007/s00170-013-5464-1


def get_Zstar_lower_bound(matrix):
    # calculating the maximum of the row/col minima
    col_min = np.min(matrix, axis=0)
    row_min = np.min(matrix, axis=1)
    col_maxmin = np.max(col_min)
    row_maxmin = np.max(row_min)
    LB = np.max([row_maxmin, col_maxmin])
    return LB


def get_Zstar_upper_bound(matrix):
    shape = np.shape(matrix)

    # major diagonal UB
    max_diagonal_val = np.max(np.diagonal(matrix))

    # minor diagonal UB
    max_minor_diag_val = np.max(np.fliplr(matrix).diagonal())

    # UB from disqualifing largest m values
    min_choices = np.max(shape)
    matrix_descending_vals = -np.sort(-matrix.flatten())
    try:
        disqualify_UB = matrix_descending_vals[min_choices - 1]
    except IndexError:
        disqualify_UB = matrix_descending_vals[0, min_choices - 1]

    UB = min(max_diagonal_val, max_minor_diag_val, disqualify_UB)
    return UB


def simplify_matrix_w_bounds(matrix, LB, UB):
    # making matrix square if it is rectangular
    new_matrix = copy.deepcopy(matrix)
    for i in range(len(new_matrix)):
        for j in range(len(new_matrix[0])):
            try:
                matval = matrix[i, j]
            except IndexError:
                matval = LB
            if matval > UB:
                new_matrix[i, j] = UB
            elif matval < LB:
                new_matrix[i, j] = LB
            else:
                new_matrix[i, j] = matval
    return new_matrix


def transform_cost_matrix_into_integers(matrix):
    unique_vals, unique_inverse, unique_counts = np.unique(
        matrix, return_inverse=True, return_counts=True
    )
    new_vals = np.array(range(1, len(unique_vals) + 1))
    return (
        new_vals[unique_inverse].reshape(np.shape(matrix)),
        unique_inverse,
        unique_counts,
    )


def convert_ranked_to_f(ranked_mat, Snorm_array, unique_inverse):
    # calculating gamma
    gamma = 0
    g_sum = 0
    while g_sum < 2 * len(ranked_mat) - 1:
        g_sum += Snorm_array[gamma]
        gamma += 1

    # calculating alpha and beta
    alpha = 0
    a_sum = 0
    while a_sum < len(ranked_mat) - 1:
        a_sum += Snorm_array[alpha]
        alpha += 1

    beta = alpha * (len(ranked_mat) - 1)
    if alpha > 1:
        t_sum = 0
        S_sum = 0
        for t in range(1, alpha):
            t_sum += t * Snorm_array[t - 1]
            S_sum += Snorm_array[t - 1]
        beta += t_sum - alpha * S_sum

    # calculating the first delta f_vals
    f_vals = np.zeros(len(Snorm_array))
    for t in range(1, 1 + gamma):
        f_vals[t - 1] = t

    # calculating the remaining f values
    overflow_divide_val = 1e-100
    for t in range(1 + gamma, 1 + len(f_vals)):
        complete = False
        nan_count = 0
        while not complete:
            sigma = t - 1
            si_sum = 0
            while si_sum < len(ranked_mat):
                si_sum += Snorm_array[sigma - 1]
                if si_sum < len(ranked_mat):
                    sigma -= 1

            theta = len(ranked_mat) * f_vals[sigma - 1]
            for k in range(sigma + 1, t):
                theta += Snorm_array[k - 1] * (f_vals[k - 1] - f_vals[sigma - 1])
            if np.isnan(theta) or np.isinf(theta):
                nan_count += 1
                if nan_count > 3:
                    return np.zeros(np.shape(ranked_mat)), 1
                f_vals = f_vals * overflow_divide_val
            else:
                complete = True

        f_vals[t - 1] = theta - (beta - 1) * (overflow_divide_val**nan_count)

    f_matrix = f_vals[unique_inverse].reshape(np.shape(ranked_mat))
    return f_matrix, 0


def convert_LBAP_to_LSAP(matrix):
    """
    This algorithm converts a linear bottleneck assignment
    problem (LBAP) into a linear sum assignment problem
    (LSAP), which can be then solved by the Hungarian
    algorithm.

    Taken from [Kuo et al., Int J Adv Manuf Technol 71 (2014)](https://doi.org/10.1007/s00170-013-5464-1)
    """
    # estimate lower and upper bounds
    mat_copy = copy.deepcopy(matrix)
    LB = get_Zstar_lower_bound(matrix)
    UB = get_Zstar_upper_bound(matrix)
    # simplify the matrix according to the lower and upper bounds
    simp_mat = simplify_matrix_w_bounds(matrix, LB, UB)
    # transform the simplified matrix into a ranked matrix
    ranked_mat, unique_inverse, Snorm_array = transform_cost_matrix_into_integers(
        simp_mat
    )
    # convert the ranked matrix into an equivalent LSAP problem
    f_mat, fail_flag = convert_ranked_to_f(ranked_mat, Snorm_array, unique_inverse)
    return f_mat, LB, mat_copy, fail_flag
