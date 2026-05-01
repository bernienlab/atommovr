"""
This file contains all helper functions for the inside out algorithm
"""

import copy
import numpy as np
from typing import Callable
from collections import deque
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

from atommovr.utils.Move import Move
from atommovr.utils.AtomArray import AtomArray
from atommovr.utils.move_utils import move_atoms
from atommovr.algorithms.source.Hungarian_works import generate_AOD_cmds


def perimeter_coords(
    top: int, left: int, bottom: int, right: int
) -> list[tuple[int, int]]:
    """
    Return a list of (row, col) coordinates around the perimeter of the rectangle
    defined by (top, left, bottom, right).
    """
    coords = []
    coords.extend((top, c) for c in range(left, right + 1))  # Top row: left → right
    coords.extend(
        (r, right) for r in range(top + 1, bottom + 1)
    )  # Right column: top+1 → bottom
    coords.extend(
        (bottom, c) for c in range(right - 1, left - 1, -1)
    )  # Bottom row: right-1 → left
    coords.extend(
        (r, left) for r in range(bottom - 1, top, -1)
    )  # Left column: bottom-1 → top+1
    return coords


def def_boundary(layer_factor: int, array_len: int):
    n = array_len
    i = (n + 1) % 2  # If n is even, define the "center" as (n//2 - 1, n//2 - 1).

    center = n // 2 - i
    top = center - layer_factor
    left = center - layer_factor
    bottom = center + layer_factor + i
    right = center + layer_factor + i

    return top, left, bottom, right


def clean_empty_moves(arrays: AtomArray, move_list: list):
    op_arrays = copy.deepcopy(arrays)
    non_empty_move_list = []
    for move in move_list:
        arrays_prev = copy.deepcopy(op_arrays)
        op_arrays.move_atoms(move)
        if not atom_arrays_equal(op_arrays, arrays_prev):
            non_empty_move_list.append(move)
    return non_empty_move_list


def atom_arrays_equal(op_arrays: AtomArray, arrays_prev: AtomArray):
    return np.array_equal(
        op_arrays.matrix[:, :, 0], arrays_prev.matrix[:, :, 0]
    ) and np.array_equal(op_arrays.matrix[:, :, 1], arrays_prev.matrix[:, :, 1])


"""
Generate information for dual-species atom arrays. Inside out algo cares about:
1. out_assign & in_assign: a list of (atom, vacancy) pairs
2-1. out_path & in_path: the direct path between each (atom, vacancy) pairs
2-2. categ_2_pair: the (atom, vacancy) pairs which do not have direct path between them
"""


def gen_dual_assign_new(arrays: AtomArray, layer_factor: int):
    """
    Generate the pairings (assignments) for the inside-out rearrangement
    for both Rb and Cs species, including usage of the 'reservoir'.
    """
    prepared_assignments = []

    # 1. Get layer-based source/target coords
    Rb_source_layer = collect_coords(
        arrays, layer_factor, "layer", is_rb_source(arrays)
    )
    Cs_source_layer = collect_coords(
        arrays, layer_factor, "layer", is_cs_source(arrays)
    )
    Rb_target_layer = collect_coords(
        arrays, layer_factor, "layer", is_rb_target(arrays)
    )
    Cs_target_layer = collect_coords(
        arrays, layer_factor, "layer", is_cs_target(arrays)
    )

    # 2. Get outer-based source/target coords + reservoir
    Rb_source = collect_coords(arrays, layer_factor, "outer", is_rb_source(arrays))
    Cs_source = collect_coords(arrays, layer_factor, "outer", is_cs_source(arrays))
    Rb_target = collect_coords(arrays, layer_factor, "outer", is_rb_target(arrays))
    Cs_target = collect_coords(arrays, layer_factor, "outer", is_cs_target(arrays))
    reservoir = collect_coords(arrays, layer_factor, "outer", is_reservoir(arrays))

    # 3. Assign for Rb
    rb_assignments, reservoir = assign_species(
        Rb_source_layer, Rb_source, Rb_target_layer, Rb_target, reservoir
    )
    prepared_assignments.extend(rb_assignments)

    # 4. Assign for Cs
    cs_assignments, reservoir = assign_species(
        Cs_source_layer, Cs_source, Cs_target_layer, Cs_target, reservoir
    )
    prepared_assignments.extend(cs_assignments)

    # 5. Separate move out assignments and move in assignments
    out_assign, in_assign = separate_assign(arrays, prepared_assignments, layer_factor)

    return out_assign, in_assign


def collect_coords(
    arrays: AtomArray,
    layer_factor: int,
    region: str,
    condition_func: Callable[[int, int], bool],
) -> list[tuple[int, int]]:
    """
    Args:
        arrays (AtomArray): dual-species atom array object.
        layer_factor (int): Factor for computing boundary via def_boundary.
        region (str): Either 'layer' or 'outer'.
        condition_func (Callable): A function taking (row, col) -> bool.

    Returns:
        list[tuple[int,int]]: All (row, col) that pass the condition_func.
    """
    n = arrays.matrix.shape[0]
    top, left, bottom, right = def_boundary(layer_factor - 1, n)

    # Decide which set of candidate coordinates to gather
    if region == "layer":
        candidates = perimeter_coords(top, left, bottom, right)
        out_bound = False
    elif region == "outer":
        candidates = [
            (r, c)
            for r in range(n)
            for c in range(n)
            if out_bound_ex(r, c, top, left, bottom, right)
        ]
        out_bound = True
    elif region == "all":
        candidates = [(r, c) for r in range(n) for c in range(n)]
        out_bound = False
    else:
        raise ValueError(f"Unrecognized region: {region}")

    # Filter by the user-provided condition
    return [(r, c) for (r, c) in candidates if condition_func(r, c, out_bound)]


def is_rb_source(arrays: AtomArray) -> Callable[[int, int], bool]:
    def _check(r, c, out_bound: bool = False):
        return bool(
            arrays.matrix[r, c, 0] == 1 and (arrays.target[r, c, 0] == 0 or out_bound)
        )

    return _check


def is_rb_target(arrays: AtomArray) -> Callable[[int, int], bool]:
    def _check(r, c, out_bound: bool = False):
        return bool(arrays.target[r, c, 0] == 1 and np.sum(arrays.matrix[r, c, :]) == 0)

    return _check


def is_cs_source(arrays: AtomArray) -> Callable[[int, int], bool]:
    def _check(r, c, out_bound: bool = False):
        return bool(
            arrays.matrix[r, c, 1] == 1 and (arrays.target[r, c, 1] == 0 or out_bound)
        )

    return _check


def is_cs_target(arrays: AtomArray) -> Callable[[int, int], bool]:
    def _check(r, c, out_bound: bool = False):
        return bool(arrays.target[r, c, 1] == 1 and np.sum(arrays.matrix[r, c, :]) == 0)

    return _check


def is_reservoir(arrays: AtomArray) -> Callable[[int, int], bool]:
    """a reservoir is an empty cell in both matrix and target outside bound."""

    def _check(r, c, out_bound: bool = False):
        return bool(
            np.sum(arrays.matrix[r, c, :]) == 0 and np.sum(arrays.target[r, c, :]) == 0
        )

    return _check


def is_site_correct(arrays: AtomArray) -> Callable[[int, int], bool]:
    def _check(r, c):
        return bool(
            arrays.matrix[r, c, 0] == arrays.target[r, c, 0]
            and arrays.matrix[r, c, 1] == arrays.target[r, c, 1]
        )

    return _check


def out_bound_ex(x, y, top, left, bottom, right):
    return x < top or x > bottom or y < left or y > right


# def out_bound_in(x, y, top, left, bottom, right):
#     return x <= top or x >= bottom or y <= left or y >= right


def assign_species(
    source_layer, source_outer, target_layer, target_outer, reservoir
) -> "tuple[list[tuple[tuple[int, int]]], list[tuple]]":
    # Make move out assignments
    assignments = []
    # all_target = target_layer + target_outer + reservoir
    all_source = source_layer + source_outer

    cost_matrix = generate_cost_matrix_inside_out(all_source, target_layer)
    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    for i, j in zip(row_idx, col_idx, strict=True):
        assignments.append((all_source[i], target_layer[j]))

    return assignments, reservoir


def generate_cost_matrix_inside_out(
    layer_source_target, outer_source_target
) -> np.ndarray:
    source_positions, target_positions = layer_source_target, outer_source_target

    # Generate the cost matrix
    cost_matrix_penalty = np.zeros((len(source_positions), len(target_positions)))

    # Calculate the distance between source and target
    for i, source in enumerate(source_positions):
        for j, target in enumerate(target_positions):
            cost_matrix_penalty[i, j] = np.linalg.norm(
                np.array(source) - np.array(target)
            )

    return cost_matrix_penalty


def assign_species_old(
    source_layer, source_outer, target_layer, target_outer, reservoir
) -> "tuple[list[tuple[tuple[int, int]]], list[tuple]]":
    source_positions = source_layer + source_outer
    target_positions = target_layer + target_outer + reservoir
    cost_matrix = generate_cost_matrix_penalty(
        source_layer, source_outer, target_layer, target_outer, reservoir
    )
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    assignments = []
    for i, j in zip(row_idx, col_idx, strict=True):
        if source_positions[i] in source_layer or target_positions[j] in target_layer:
            assignments.append((source_positions[i], target_positions[j]))
        # remove from reservoir if used
        if target_positions[j] in reservoir:
            reservoir.remove(target_positions[j])
    return assignments, reservoir


## This code is no longer used ##
def generate_cost_matrix_penalty(
    source_layer, source_outer, target_layer, target_outer, reservoir
) -> np.ndarray:
    source_positions, target_positions = (
        source_layer + source_outer,
        target_layer + target_outer + reservoir,
    )
    layer_count_s, layer_count_t = len(source_layer), len(target_layer)

    # Generate the cost matrix
    cost_matrix_penalty = np.zeros((len(source_positions), len(target_positions)))

    # Calculate the distance between source and target
    for i, source in enumerate(source_positions):
        for j, target in enumerate(target_positions):
            if (i < layer_count_s and j < layer_count_t) and np.linalg.norm(
                np.array(source) - np.array(target)
            ) <= 1:
                cost_matrix_penalty[i, j] = -10000
            elif i < layer_count_s or j < layer_count_t:
                cost_matrix_penalty[i, j] = np.linalg.norm(
                    np.array(source) - np.array(target)
                )
            else:
                cost_matrix_penalty[i, j] = 10000

    return cost_matrix_penalty


def separate_assign(arrays: AtomArray, prepared_assignments: list, layer_factor: int):
    out_assign, in_assign = [], []
    top, left, bottom, right = def_boundary(layer_factor - 1, arrays.matrix.shape[0])
    layer_coords = perimeter_coords(top, left, bottom, right)

    for source, target in prepared_assignments:
        if source in layer_coords:
            out_assign.append((source, target))
        else:
            in_assign.append((source, target))

    return out_assign, in_assign


"""
Generating path part of code
"""


@dataclass
class BFSResult:
    path: list[tuple[int, int]]
    end_reached: bool
    same_obstacle: list[tuple[int, int]] | None
    diff_obstacle: list[tuple[int, int]] | None
    category: int


def process_chain_moves_new(bfs_res: BFSResult):
    bfs_res.same_obstacle = (
        []
        if bfs_res.same_obstacle is None
        else bfs_res.same_obstacle  # previously == None
    )
    single_path = []
    segmant_path = []
    for coord in bfs_res.path:
        segmant_path.append(coord)
        if coord in bfs_res.same_obstacle:
            single_path.append(tuple(segmant_path))
            segmant_path = [coord]
    single_path.append(tuple(segmant_path))
    return single_path[::-1]


def generate_decomposed_move_list(
    op_arrays: AtomArray, single_path: list, move_list_for_assigns: list
):
    # Iterate all path segments (((a1,b1), (a2,b2), (a3, b3), (a4,b4)), ((c1,d1), (c2,d2)))
    for segmant in single_path:
        segmant_moves = []
        from_row, from_col = segmant[0]
        if len(segmant) <= 1:
            continue
        op_arrays.matrix[from_row][from_col] = 0
        op_arrays.matrix[segmant[1][0]][segmant[1][0]] = 1
        for coordinate in segmant:
            to_row, to_col = coordinate
            if (to_row, to_col) != (from_row, from_col):  # To exclude the frist move
                segmant_moves.append(Move(from_row, from_col, to_row, to_col))
                from_row, from_col = to_row, to_col
        move_list_for_assigns.append(segmant_moves)
    return move_list_for_assigns


def generate_decomposed_move_list_old(
    op_arrays: AtomArray, single_path: list, move_list_for_assigns: list
):
    # Iterate all path segments (((a1,b1), (a2,b2), (a3, b3), (a4,b4)), ((c1,d1), (c2,d2)))
    for segmant in single_path:
        segmant_moves = []
        from_row, from_col = segmant[0]
        for coordinate in segmant:
            to_row, to_col = coordinate
            if (to_row, to_col) != (from_row, from_col):  # To exclude the frist move
                segmant_moves.append(Move(from_row, from_col, to_row, to_col))
                op_arrays.matrix[from_row][from_col] = 0
                op_arrays.matrix[to_row][to_col] = 1
                from_row, from_col = to_row, to_col
        move_list_for_assigns.append(segmant_moves)
    return move_list_for_assigns


def neighbors_8_ex_end(
    r: int, c: int, n: int, layer_bound: list, end: tuple
) -> list[tuple[int, int]]:
    neighbors = []
    directions = [
        (dr, dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (dr, dc) != (0, 0)
    ]
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if (
            out_bound_ex(
                nr, nc, layer_bound[0], layer_bound[1], layer_bound[2], layer_bound[3]
            )
            and 0 <= nr < n
            and 0 <= nc < n
        ) or (nr, nc) == end:
            neighbors.append((nr, nc))
    return neighbors


def neighbors_8(r: int, c: int, n: int, layer_bound: list) -> list[tuple[int, int]]:
    neighbors = []
    directions = [
        (dr, dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (dr, dc) != (0, 0)
    ]
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if (
            out_bound_in(
                nr, nc, layer_bound[0], layer_bound[1], layer_bound[2], layer_bound[3]
            )
            and 0 <= nr < n
            and 0 <= nc < n
        ):
            neighbors.append((nr, nc))
    return neighbors


def bfs_find_path_new(
    matrix: np.ndarray,
    layer_factor: int,
    start: tuple[int, int],
    end: tuple[int, int],
    handle_obstacle_filter: Callable[[tuple[int, int], tuple[int, int]], bool],
) -> BFSResult:

    n = matrix.shape[0]
    layer_bound = list(def_boundary(layer_factor - 1, n))

    # Initialize a queue for BFS. Each element is (row, col, path_so_far)
    visited = set([start])
    same_obstacle_list = []
    diff_obstacle_list = []
    queue = deque(
        [(start[0], start[1], [start], same_obstacle_list, diff_obstacle_list)]
    )

    while queue:
        row, col, path_so_far, same_obstacle_list, diff_obstacle_list = queue.popleft()

        # If we reached the end, return the path
        if (row, col) == end:
            if len(diff_obstacle_list) == 0:
                return BFSResult(
                    path_so_far, True, same_obstacle_list, diff_obstacle_list, 1
                )
            else:
                return BFSResult(
                    path_so_far, True, same_obstacle_list, diff_obstacle_list, 2
                )

        # Explore all possible neighbors
        for new_r, new_c in neighbors_8_ex_end(row, col, n, layer_bound, end):
            if (new_r, new_c) not in visited:
                pass_flag, homo_obs, hetero_obs = handle_obstacle_filter(
                    start, (new_r, new_c)
                )
                if pass_flag:
                    visited.add((new_r, new_c))
                    if homo_obs:
                        queue.append(
                            (
                                new_r,
                                new_c,
                                path_so_far + [(new_r, new_c)],
                                same_obstacle_list + [homo_obs],
                                diff_obstacle_list,
                            )
                        )
                    elif hetero_obs:
                        queue.append(
                            (
                                new_r,
                                new_c,
                                path_so_far + [(new_r, new_c)],
                                same_obstacle_list,
                                diff_obstacle_list + [hetero_obs],
                            )
                        )
                    else:
                        queue.append(
                            (
                                new_r,
                                new_c,
                                path_so_far + [(new_r, new_c)],
                                same_obstacle_list,
                                diff_obstacle_list,
                            )
                        )

    return BFSResult(path_so_far, False, same_obstacle_list, diff_obstacle_list, 2)


def direct_path_ok(
    arrays: AtomArray,
) -> Callable[
    [tuple[int, int], tuple[int, int]],
    tuple[bool, tuple[int, int] | None, tuple[int, int] | None],
]:
    def _check(start: tuple[int, int], new_site: tuple[int, int]):
        # If cell is empty => pass_flag=True
        if np.sum(arrays.matrix[new_site[0], new_site[1], :]) == 0:
            return True, None, None
        else:
            return False, None, None

    return _check


def same_species_ok(
    arrays: AtomArray,
) -> Callable[
    [tuple[int, int], tuple[int, int]],
    tuple[bool, tuple[int, int] | None, tuple[int, int] | None],
]:
    direct_func = direct_path_ok(arrays)

    def _check(start: tuple[int, int], new_site: tuple[int, int]):
        pass_flag, _, _ = direct_func(start, new_site)  # First check if it's empty
        if pass_flag:
            return True, None, None  # It's empty => BFS can proceed, no obstacle
        else:
            same_rb = (
                arrays.matrix[start[0], start[1], 0]
                == arrays.matrix[new_site[0], new_site[1], 0]
            )
            same_cs = (
                arrays.matrix[start[0], start[1], 1]
                == arrays.matrix[new_site[0], new_site[1], 1]
            )
            if same_rb and same_cs:
                return True, new_site, None
            else:
                return False, None, None

    return _check


def diff_species_ok(
    arrays: AtomArray,
) -> Callable[
    [tuple[int, int], tuple[int, int]],
    tuple[bool, tuple[int, int] | None, tuple[int, int] | None],
]:
    same_func = same_species_ok(arrays)

    def _check(start: tuple[int, int], new_site: tuple[int, int]):
        pass_flag, homo_obs, _ = same_func(start, new_site)
        if pass_flag:
            return pass_flag, homo_obs, None
        else:
            return True, None, new_site

    return _check


def out_bound_in(x, y, top, left, bottom, right):
    return y <= top or y >= bottom or x <= left or x >= right


def collect_non_conflicting_moves(
    candidates: list[Move], arrays: AtomArray
) -> list[Move]:
    """
    Given a list of candidate moves (one from each path),
    pick a subset that do not collide with each other and are valid (destination is free, etc.).
    Returns a list of moves that can be done simultaneously.
    """
    # 1) If a move doesn't conflict with what we've selected, add it.
    selected_moves = []
    used_from = set()
    used_to = set()

    # You may need a quick snapshot of your occupancy matrix
    matrix_occupancy = arrays.matrix[:, :, 0] + arrays.matrix[:, :, 1]

    for move in candidates:
        used_from.add((move.from_row, move.from_col))

    for move in candidates:
        if (move.to_row, move.to_col) in used_to:
            # collision on the destination
            continue

        if matrix_occupancy[move.to_row, move.to_col] != 0:
            # Parallize chain moves allow destination is occupied
            if (move.from_row, move.from_col) not in used_from:
                continue

        # No conflict => select it
        selected_moves.append(move)
        used_to.add((move.to_row, move.to_col))

    return selected_moves


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

            # horiz_AOD_cmds, vert_AOD_cmds, can_parallelize, move_list_with_ghost = generate_AOD_cmds(matrix_copy, parallel_moves + [p_move])
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
                matrix_copy, _ = move_atoms(matrix_copy, parallel_moves_test)
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
        matrix_copy, _ = move_atoms(matrix_copy, parallel_moves)
        total_atom_num_final = np.sum(matrix_copy)

        if total_atom_num_init == total_atom_num_final:
            parallel_seq.append(parallel_moves)
        else:
            matrix_copy = copy.deepcopy(sanit_check_matrix)
    return parallel_seq


def push_out_obstacles(
    arrays: AtomArray,
    layer_factor: int,
    obstacles: list[tuple[int, int]],
    raw_path: list[list[Move]],
) -> tuple[AtomArray, list[list[Move]]]:
    push_moves = []
    path = set()
    for seg in raw_path:
        for coord in seg:
            path.add(coord)

    for obs_coord in obstacles:
        # Find a good site to push
        push_coord, push_dir, distance = find_push_coord(
            arrays, obs_coord, layer_factor, path
        )

        if distance == 1:  # direct push
            move = Move(obs_coord[0], obs_coord[1], push_coord[0], push_coord[1])
            arrays.move_atoms([move])
            push_moves.append([move])
        else:  # chain push
            chain_list = chain_push_move(obs_coord, push_coord, push_dir)
            arrays.move_atoms(chain_list)
            push_moves.append(chain_list)

    return arrays, push_moves


def find_push_coord(
    arrays: AtomArray,
    obs_coord: tuple[int, int],  # Info for pushing object
    layer_factor: int,
    path_direction_exclusions: list[tuple[int, int]],  # Pushing constraints
) -> tuple[int, int]:
    """
    1) Identify which directions are outward.
    2) Exclude directions used by the path or that go inward.
    3) For each remaining direction, find the nearest empty site.
    """

    obs_r, obs_c = obs_coord
    n = arrays.matrix.shape[0]
    center = n // 2
    top, left, bottom, right = def_boundary(layer_factor, n)
    layer_exclusions = perimeter_coords(top, left, bottom, right)

    # 1) Build candidate directions
    candidate_dirs = []
    directions = [
        (dr, dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (dr, dc) != (0, 0)
    ]
    relaxed_condition = 0

    while not candidate_dirs:
        for dr, dc in directions:
            nr, nc = obs_r + dr, obs_c + dc
            # skip direction if the pushed destination still blocks
            if (nr, nc) in path_direction_exclusions and relaxed_condition <= 2:
                continue
            # skip direction if the pushed destination on the layer
            if (nr, nc) in layer_exclusions and relaxed_condition <= 1:
                continue
            # skip direction if it heads inward
            if (
                is_inward_direction(obs_r, obs_c, nr, nc, center, center)
                and relaxed_condition <= 0
            ):
                continue
            candidate_dirs.append((dr, dc))
        # Relax searching condition if the no pushed site available
        relaxed_condition += 1
        if relaxed_condition > 3 and not candidate_dirs:
            raise Exception(
                "No valid push direction found for obstacle at ({}, {})".format(
                    obs_r, obs_c
                )
            )

    # 2) For each direction, find the nearest empty site
    #    We'll store (found_coord, direction)
    possible_sites = []
    ejection_flag = False
    while not possible_sites:
        for dr, dc in candidate_dirs:
            site = find_empty_in_direction(arrays, obs_r, obs_c, dr, dc, ejection_flag)
            if site:
                possible_sites.append((site, dr, dc))
        ejection_flag = True  # Allow ejecting atom if no empty site available

    # 3) Among possible sites, pick the one with largest distance from center
    best_site = None
    best_dist = 1e6
    for site, dr, dc in possible_sites:
        r, c = site
        dist_sq = (((r - obs_r) ** 2 + (c - obs_c) ** 2) / (dr**2 + dc**2)) ** 0.5
        if dist_sq < best_dist:
            best_dist = dist_sq
            best_site = site
            push_dir = (dr, dc)

    return best_site, push_dir, dist_sq


def find_empty_in_direction(
    arrays: AtomArray, obs_r: int, obs_c: int, dr: int, dc: int, ejection: bool
) -> tuple[int, int] | None:
    n = arrays.matrix.shape[0]
    cur_r, cur_c = obs_r, obs_c

    while True:
        cur_r += dr
        cur_c += dc
        if cur_r < 0 or cur_r >= n or cur_c < 0 or cur_c >= n:
            if not ejection:
                return None
            else:
                return (cur_r, cur_c)
        # If cell is empty
        if np.sum(arrays.matrix[cur_r, cur_c, :]) == 0:
            return (cur_r, cur_c)


def is_inward_direction(obs_r, obs_c, nr, nc, center_r, center_c) -> bool:
    # If new_dist < old_dist, it means heading inward
    return (nr - center_r) ** 2 + (nc - center_c) ** 2 < (obs_r - center_r) ** 2 + (
        obs_c - center_c
    ) ** 2


def chain_push_move(obs_coord, push_coord, push_dir):
    chain_list = []
    cur_r, cur_c = obs_coord
    dr, dc = push_dir
    while True:
        new_r, new_c = cur_r + dr, cur_c + dc
        chain_list.append((Move(cur_r, cur_c, new_r, new_c)))
        if (new_r, new_c) == push_coord:
            break
        cur_r, cur_c = new_r, new_c
    return chain_list[::-1]


def categ_2_move_exe(arrays: AtomArray, single_path: list, all_categ2_moves: list):
    op_arrays = copy.deepcopy(arrays)
    main_moves = generate_decomposed_move_list(
        op_arrays, single_path, move_list_for_assigns=[]
    )
    for seg in main_moves:
        for move in seg:
            all_categ2_moves.append([move])
            arrays.move_atoms([move])
    return arrays, all_categ2_moves
