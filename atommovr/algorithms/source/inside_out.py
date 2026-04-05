import copy
import numpy as np
from typing import Callable

from atommovr.utils.AtomArray import AtomArray
from atommovr.utils.Move import Move
from atommovr.algorithms.source.inside_out_utils import (
    is_site_correct,
    clean_empty_moves,
    def_boundary,
    perimeter_coords,
    collect_coords,
    is_rb_source,
    is_cs_source,
    chain_push_move,
    find_empty_in_direction,
    bfs_find_path_new,
    generate_AOD_cmds,
    same_species_ok,
    process_chain_moves_new,
    collect_non_conflicting_moves,
    categ_2_move_exe,
    push_out_obstacles,
    diff_species_ok,
    regroup_parallel_moves,
    gen_dual_assign_new,
    generate_decomposed_move_list,
)


def inside_out_algorithm(
    rbcs_arrays: AtomArray,
    do_ejection: bool = False,
    round_lim: int = 50,
):
    """
    Inside out rearrangement algorithm proposed by the Bernien lab at the University of Chicago.

    ``do_ejection`` is accepted for API compatibility with algorithm wrappers.
    The current InsideOut implementation does not perform an ejection stage.
    """
    arrays = copy.deepcopy(rbcs_arrays)
    move_list = []
    move_list_layer = []

    if not check_atom_enough(rbcs_arrays):  # == False: linting error
        return rbcs_arrays, [], False

    layer_num = 1  # The index control which layer we are rearranging
    iteration = 0

    # Start deal with the n-th layer rearrangement
    while (
        not rearrangement_complete(arrays) and iteration < round_lim
    ):  # == False linting error
        arrays_save = copy.deepcopy(arrays)
        if layer_complete(layer_num, arrays, is_site_correct(arrays)):
            move_list.extend(move_list_layer)
            move_list_layer = []
            # If the rearrangement is not completed, we move on next outer layer
            if not rearrangement_complete(arrays):
                layer_num += 1
        else:
            if np.array_equal(arrays.matrix, arrays_save.matrix) and iteration > 0:
                arrays, moves = inside_out_layer_push(
                    arrays, layer_num, get_stuck_flag=True
                )
            else:
                arrays, moves = inside_out_layer_push(
                    arrays, layer_num, get_stuck_flag=False
                )

            if moves is None:  # == None: linting error
                return rbcs_arrays, None, False
            move_list_layer.extend(moves)
            iteration += 1

            # Check if the rearrangement disturbs inner (previous) layer
            for check_layer in range(layer_num, 0, -1):
                if not layer_complete(check_layer, arrays, is_site_correct(arrays)):
                    layer_num = check_layer

    if rearrangement_complete(arrays):
        move_list.extend(move_list_layer)

    return (
        arrays,
        clean_empty_moves(rbcs_arrays, move_list),
        rearrangement_complete(arrays),
    )


def check_atom_enough(arrays: AtomArray) -> bool:
    """
    Check if the initial arrays have enough atoms to rearrange.
    """
    Rb_num = np.sum(arrays.matrix[:, :, 0])
    Cs_num = np.sum(arrays.matrix[:, :, 1])
    Rb_target_num = np.sum(arrays.target_Rb)
    Cs_target_num = np.sum(arrays.target_Cs)

    if Rb_num >= Rb_target_num and Cs_num >= Cs_target_num:
        return True
    else:
        return False


def layer_complete(
    layer_factor: int, arrays: AtomArray, is_site_correct: Callable[[int, int], bool]
) -> bool:
    """
    Check if the layer of an atom array is complete for both Rb and Cs.
    A layer is 'complete' if, along the perimeter of this layer,
    the array matches the target for both Rb and Cs.
    """
    # Extract Rb & Cs layers plus their target arrays
    Rb_arrays = arrays.matrix[:, :, 0]
    Rb_target = arrays.target_Rb
    Cs_arrays = arrays.matrix[:, :, 1]
    Cs_target = arrays.target_Cs

    # Define the boundary for this layer
    n = Rb_arrays.shape[0]
    top, left, bottom, right = def_boundary(layer_factor - 1, n)

    # Boundary check
    if top < 0 or left < 0 or bottom >= n or right >= n:
        return False

    # If layer_factor == 0, it's just the center cell
    if top == bottom and left == right:
        return bool(
            Rb_arrays[top, left] == Rb_target[top, left]
            and Cs_arrays[top, left] == Cs_target[top, left]
        )

    # Check perimeter cells; if any mismatch, layer isn't complete
    for r, c in perimeter_coords(top, left, bottom, right):
        if not is_site_correct(r, c):
            return False

    return True


def inside_out_layer_push(
    rbcs_arrays: AtomArray, layer_factor: int, get_stuck_flag: bool
):
    arrays = copy.deepcopy(rbcs_arrays)
    move_list = []

    # Push out all misplace atoms
    arrays, push_out_moves = push_out_misplaced_atoms(arrays, layer_factor)
    move_list.extend(push_out_moves)

    if not check_atom_enough(arrays):  # == False: linting error
        return arrays, None

    # Check if the process get stuck
    while True:
        N_independent_path_move_in, categ_2_pair_in = generate_path_inside_out_new(
            arrays, layer_factor, "in"
        )

        if len(N_independent_path_move_in) == 0:
            break
        if get_stuck_flag:
            for move in N_independent_path_move_in[0]:
                arrays.move_atoms([move])
                move_list.append([move])
        else:
            arrays, categ_1_moves_in = transform_paths_into_moves(
                arrays, N_independent_path_move_in, max_rounds=1
            )
            move_list.extend(categ_1_moves_in)

    if len(categ_2_pair_in) > 0:
        arrays, all_categ_2_moves = handle_categ_2_paths(
            arrays, categ_2_pair_in, layer_factor
        )
        move_list.extend(all_categ_2_moves)

    return arrays, move_list


def rearrangement_complete(arrays: AtomArray) -> bool:
    """
    It should call the layer_complete function (every layer complete->rearrangement complete)
    """
    Rb_complete = np.array_equal(
        np.multiply(arrays.matrix[:, :, 0], arrays.target_Rb), arrays.target_Rb
    )
    Cs_complete = np.array_equal(
        np.multiply(arrays.matrix[:, :, 1], arrays.target_Cs), arrays.target_Cs
    )
    return bool(Rb_complete and Cs_complete)


def push_out_misplaced_atoms(arrays: AtomArray, layer_factor: int):
    # 1. Get the coordinate of misplaced atoms
    Rb_source_layer = collect_coords(
        arrays, layer_factor, "layer", is_rb_source(arrays)
    )
    Cs_source_layer = collect_coords(
        arrays, layer_factor, "layer", is_cs_source(arrays)
    )
    # Rb_target_layer = collect_coords(
    #     arrays, layer_factor, "layer", is_rb_target(arrays)
    # )
    # Cs_target_layer = collect_coords(
    #     arrays, layer_factor, "layer", is_cs_target(arrays)
    # )

    # 2. Push out all misplaced atoms
    arrays, push_moves = crude_push_atoms(
        arrays, layer_factor, Rb_source_layer, Cs_source_layer
    )

    return arrays, push_moves


def crude_push_atoms(
    arrays: AtomArray,
    layer_factor: int,
    Rb_source_layer: list[tuple[int, int]],
    Cs_source_layer: list[tuple[int, int]],
) -> tuple[AtomArray, list[list[Move]]]:
    push_moves_dict = {(1, 0): [], (0, 1): [], (-1, 0): [], (0, -1): []}
    non_parallel_push_moves_dict = {(1, 0): [], (0, 1): [], (-1, 0): [], (0, -1): []}
    push_moves = []
    op_matrix = copy.deepcopy(arrays.matrix[:, :, 0] + arrays.matrix[:, :, 1])

    misplaced_atoms = Rb_source_layer + Cs_source_layer

    for obs_coord in misplaced_atoms:
        # Find a good site to push
        push_coord, push_dir, distance = find_push_coord_misplaced(
            arrays, obs_coord, layer_factor
        )

        if distance == 1:  # direct push
            move = Move(obs_coord[0], obs_coord[1], push_coord[0], push_coord[1])
            arrays.move_atoms([move])
            # push_moves.append([move])
            push_moves_dict[(push_dir[0], push_dir[1])].extend([move])
            non_parallel_push_moves_dict[(push_dir[0], push_dir[1])].append([move])
        else:  # chain push
            chain_list = chain_push_move(obs_coord, push_coord, push_dir)
            arrays.move_atoms(chain_list)
            # push_moves.append(chain_list)
            push_moves_dict[(push_dir[0], push_dir[1])].extend(chain_list)
            non_parallel_push_moves_dict[(push_dir[0], push_dir[1])].append(chain_list)

    if len(push_moves_dict[(1, 0)]) > 0:
        horiz_AOD_cmds, vert_AOD_cmds, parallel_success_flag = generate_AOD_cmds(
            op_matrix, push_moves_dict[(1, 0)]
        )
        if parallel_success_flag:
            push_moves.append(push_moves_dict[(1, 0)])
        else:
            for push_line in non_parallel_push_moves_dict[(1, 0)]:
                push_moves.append(push_line)

    if len(push_moves_dict[(0, 1)]) > 0:
        horiz_AOD_cmds, vert_AOD_cmds, parallel_success_flag = generate_AOD_cmds(
            op_matrix, push_moves_dict[(0, 1)]
        )
        if parallel_success_flag:
            push_moves.append(push_moves_dict[(0, 1)])
        else:
            for push_line in non_parallel_push_moves_dict[(0, 1)]:
                push_moves.append(push_line)

    if len(push_moves_dict[(-1, 0)]) > 0:
        horiz_AOD_cmds, vert_AOD_cmds, parallel_success_flag = generate_AOD_cmds(
            op_matrix, push_moves_dict[(-1, 0)]
        )
        if parallel_success_flag:
            push_moves.append(push_moves_dict[(-1, 0)])
        else:
            for push_line in non_parallel_push_moves_dict[(-1, 0)]:
                push_moves.append(push_line)

    if len(push_moves_dict[(0, -1)]) > 0:
        horiz_AOD_cmds, vert_AOD_cmds, parallel_success_flag = generate_AOD_cmds(
            op_matrix, push_moves_dict[(0, -1)]
        )
        if parallel_success_flag:
            push_moves.append(push_moves_dict[(0, -1)])
        else:
            for push_line in non_parallel_push_moves_dict[(0, -1)]:
                push_moves.append(push_line)

    return arrays, push_moves


def find_target_neighbor(source_layer, target_layer):
    dir = [
        (dr, dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (abs(dr) + abs(dc)) == 1
    ]
    for dr, dc in dir:
        if (source_layer[0] + dr, source_layer[1] + dc) in target_layer:
            return (source_layer[0] + dr, source_layer[1] + dc), (dr, dc)
    return None, None


def find_push_coord_misplaced(
    arrays: AtomArray, obs_coord: tuple[int, int], layer_factor: int
) -> tuple[int, int]:
    """
    1) Identify which directions are outward.
    2) Exclude directions used by the path or that go inward.
    3) For each remaining direction, find the nearest empty site.
    """
    obs_r, obs_c = obs_coord

    # 1) Build candidate directions
    push_dir = find_push_dir(arrays, layer_factor, obs_coord)

    # 2) For each direction, find the nearest empty site. We'll store (found_coord, direction)
    ejection_flag = True
    dr, dc = push_dir[0], push_dir[1]
    site = find_empty_in_direction(arrays, obs_r, obs_c, dr, dc, ejection_flag)

    dist_sq = (
        ((site[0] - obs_r) ** 2 + (site[1] - obs_c) ** 2) / (dr**2 + dc**2)
    ) ** 0.5

    return site, push_dir, dist_sq


"""
Functions for executing inside_out_layer
"""


def generate_path_inside_out_new(
    arrays: AtomArray, layer_factor: int, in_or_out: str
) -> list:
    """
    For each (start, end), run BFS to find a path. If BFS fails or finds different-species occupant, log to type_2_pair. Return a list of (move_list, category).
    """
    move_list_for_assigns = []
    categ_2_pair = []
    op_arrays = copy.deepcopy(arrays)
    out_assign, in_assign = gen_dual_assign_new(arrays, layer_factor)
    prepared_assignments = out_assign + in_assign

    for start, end in prepared_assignments:
        bfs_res = bfs_find_path_new(
            op_arrays.matrix, layer_factor, start, end, same_species_ok(op_arrays)
        )
        if bfs_res.end_reached:
            single_path = process_chain_moves_new(bfs_res)
            move_list_for_assigns = generate_decomposed_move_list(
                op_arrays, single_path, move_list_for_assigns
            )
        else:
            categ_2_pair.append((start, end))
    return move_list_for_assigns, categ_2_pair


def transform_paths_into_moves(
    arrays: AtomArray, all_paths: list[list[Move]], max_rounds: int = 1
) -> tuple[AtomArray, list[list[Move]]]:
    """
    Execute up to one move from each path in 'paths' per round, avoiding collisions. Collisions occur if two moves share a 'to' or 'from' coordinate.
    Returns: (arrays, parallel_moves)
    parallel_moves: list of rounds, each round is a list of Move objects that were executed simultaneously.
    """
    parallel_moves = []
    round_count = 0

    """If max_rounds set 1, we could dynamically search newest path every time."""
    while round_count < max_rounds:
        # 1) Gather the candidate move from each path, if available
        move_candidates = []
        for path in all_paths:
            if len(path) > 0:  # path is not empty
                move_candidates.append(path[0])  # next move in this path

        # 2) Identify non-conflicting moves among 'candidates'
        moves_in_scan = collect_non_conflicting_moves(move_candidates, arrays)

        # 3) Parallelize moves in this round
        if len(moves_in_scan) > 0:
            matrix = arrays.matrix[:, :, 0] + arrays.matrix[:, :, 1]
            moves_in_scan = regroup_parallel_moves(matrix, moves_in_scan)
            # 2.1.3 Implement the moves
            parallel_moves.extend(moves_in_scan)
            for moves in moves_in_scan:
                _ = arrays.move_atoms(moves)

        # 4) Remove them from each path
        for move in moves_in_scan:
            for path in all_paths:
                if path and path[0] == move:
                    path.pop(0)  # remove this move from that path
                    break
        round_count += 1

    return arrays, parallel_moves


def handle_categ_2_paths(
    arrays: AtomArray, categ_2_pairs: list[tuple[int, int]], layer_factor: int
) -> tuple[AtomArray, list[list[Move]]]:
    """
    For each 'categ_2' source-target pair (start, end):
      1) Use BFS with same_species_ok to find a path. If no result, use diff_species_ok to find a path.
      2) If BFS finds different-species obstacles, push them aside (crude_push_new).
      3) Convert the final BFS path to a list of moves, apply them if desired.
    """
    all_categ2_moves = []
    op_arrays = copy.deepcopy(arrays)

    for start, end in categ_2_pairs:
        # 1) Try to search trivial path again
        bfs_res = bfs_find_path_new(
            op_arrays.matrix, layer_factor, start, end, same_species_ok(op_arrays)
        )
        if not bfs_res.end_reached:
            bfs_res_allow_diff = bfs_find_path_new(
                op_arrays.matrix, layer_factor, start, end, diff_species_ok(op_arrays)
            )
            single_path = process_chain_moves_new(bfs_res_allow_diff)
        else:
            single_path = process_chain_moves_new(bfs_res)
            op_arrays, all_categ2_moves = categ_2_move_exe(
                op_arrays, single_path, all_categ2_moves
            )
            continue

        # If BFS found obstacles (diff_obstacle)
        if bfs_res_allow_diff.diff_obstacle:
            # 2) Attempt to push them out
            op_arrays, push_moves = push_out_obstacles(
                op_arrays, layer_factor, bfs_res_allow_diff.diff_obstacle, single_path
            )
            # accumulate or log these push moves if you want
            all_categ2_moves.extend(push_moves)

        # 3) Search BFS again to obtain optimal obstacle-free path
        bfs_res = bfs_find_path_new(
            op_arrays.matrix, layer_factor, start, end, same_species_ok(op_arrays)
        )
        single_path = process_chain_moves_new(bfs_res)
        op_arrays, all_categ2_moves = categ_2_move_exe(
            op_arrays, single_path, all_categ2_moves
        )

    return op_arrays, all_categ2_moves


def find_push_dir(arrays, layer_factor, obs_coord):
    n = arrays.matrix.shape[0]
    top, left, bottom, right = def_boundary(layer_factor - 1, n)
    for c in range(left, right + 1):
        if (top, c) == obs_coord:
            return (-1, 0)
    for r in range(top + 1, bottom + 1):
        if (r, right) == obs_coord:
            return (0, 1)
    for c in range(right - 1, left - 1, -1):
        if (bottom, c) == obs_coord:
            return (1, 0)
    for r in range(bottom - 1, top, -1):
        if (r, left) == obs_coord:
            return (0, -1)
