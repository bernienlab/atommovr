# Core functions and classes for moving atoms
from __future__ import annotations
import copy
import numpy as np
from enum import IntEnum
from collections import Counter
from numpy.typing import NDArray
from typing import TypeAlias

from atommovr.utils.Move import Move
from atommovr.utils.ErrorModel import ErrorModel
from atommovr.utils.errormodels import ZeroNoise
from atommovr.utils.core import PhysicalParams

IntArray: TypeAlias = NDArray[np.integer]
BoolArray: TypeAlias = NDArray[np.bool_]


class MoveType(IntEnum):
    """
    Class to be used in conjunction with `move_atoms()` to track the legality of moves
    (separately from the `FailureFlag` and `FailureEvent`) classes, which track error
    events.
    """

    ILLEGAL_MOVE = 0
    LEGAL_MOVE = 1
    EJECT_MOVE = 2
    NO_ATOM_TO_MOVE = 3


class MultiOccupancyFlag(IntEnum):
    """
    Post-apply annotation for array-level occupancy violations.

    This exists to help users quantify and diagnose deterministic “cleanup” effects
    that happen *after* move outcomes have been applied (e.g., multi-atom occupancy
    of a single optical tweezer after a batch update).

    Notes
    -----
    This is intentionally separate from `FailureEvent`/`FailureFlag`, which describe
    single-move outcomes determined before `_apply_moves(...)` mutates the array.
    """

    SINGLE_OR_NO_OCCUPANCY = 0
    MULTI_ATOM_OCCUPANCY = 1


## Event mask (for vectorization) ##
def alloc_event_mask(n_moves: int, *, dtype=np.uint64) -> NDArray:
    """
    Allocate a per-move event mask array.

    Why this exists
    ---------------
    The failure pipeline represents multiple failure mechanisms as a bitmask per move.
    This helper centralizes allocation and makes the intended dtype explicit.

    Parameters
    ----------
    n_moves
        Number of moves to allocate for.
    dtype
        Integer dtype for the bitmask container. `np.uint64` is recommended.

    Returns
    -------
    event_mask
        Zero-initialized array of shape (n_moves,).
    """
    return np.zeros(int(n_moves), dtype=dtype)


## AOD cmd functions ##

AOD_cmd_to_pos_shift = {"1": 0, "2": 1, "3": -1}


def get_move_list_from_AOD_cmds(
    horiz_AOD_cmds: list[int] | NDArray[np.integer],
    vert_AOD_cmds: list[int] | NDArray[np.integer],
) -> list[Move]:
    """
    Convert AOD command vectors into a list of Moves.

    Design intent
    -------------
    This is a lightweight, geometry-only decoder that turns axis commands into a
    set of intended atom transports. It does **not** consult occupancy and does not
    attempt to resolve conflicts (crossings, collisions, duplicates).

    Semantics
    ---------
    - AOD commands are per-axis integers in {0,1,2,3}.
    - 0 = tone off (no move contribution)
    - 1 = static / hold
    - 2 = move +1 along that axis
    - 3 = move -1 along that axis
    - A move is emitted for each (row, col) where both row_cmd and col_cmd are nonzero
      and at least one axis is moving (i.e. row_cmd * col_cmd != 1).

    Parameters
    ----------
    horiz_AOD_cmds, vert_AOD_cmds
        Horizontal (columns) and vertical (rows) command vectors.

    Returns
    -------
    move_list
        List of Move objects implied by the axis commands.
    """
    move_list = []
    for row_ind in range(len(vert_AOD_cmds)):
        row_cmd = int(vert_AOD_cmds[row_ind])
        if row_cmd == 0:
            continue
        row_shift = AOD_cmd_to_pos_shift["{}".format(row_cmd)]
        for col_ind in range(len(horiz_AOD_cmds)):
            col_cmd = int(horiz_AOD_cmds[col_ind])
            if col_cmd != 0:  # and col_cmd*row_cmd != 1:
                # make a move
                col_shift = AOD_cmd_to_pos_shift["{}".format(col_cmd)]
                move_list.append(
                    Move(row_ind, col_ind, row_ind + row_shift, col_ind + col_shift)
                )
    return move_list


def get_AOD_cmds_from_move_list(
    matrix: NDArray, move_seq: list[Move], verify: bool = False
) -> tuple[NDArray[np.int8], NDArray[np.int8], bool]:
    """
    Infer AOD command vectors implied by a proposed parallel move set.

    Why this exists
    ---------------
    Some algorithm implementations reason about and/or cache AOD command vectors
    rather than working with Move lists directly. This function produces a
    (horiz_cmds, vert_cmds) representation *if* the move set is compatible with a
    single command per source row/col.

    Key rule: per-source-row/col consistency
    ---------------------------------------
    For a set of moves to be representable as axis commands:
    - Every source row must have a single consistent vertical command (0/1/2/3).
    - Every source column must have a single consistent horizontal command (0/1/2/3).
    If two moves require conflicting commands on the same source row/col, the move
    set is not representable and `parallel_success_flag` is False.

    Parameters
    ----------
    matrix
        Occupancy matrix used only for dimensions (row/col count). Contents are not
        consulted for timing decisions here.
    move_seq
        Proposed parallel move set.

    Returns
    -------
    horiz_AOD_cmds, vert_AOD_cmds
        Axis command vectors (dtype int8) sized to the matrix shape.
    parallel_success_flag
        True if the move set can be encoded without command conflicts.
    """
    row_num = len(matrix)
    col_num = len(matrix[0])
    horiz_AOD_cmds = np.zeros([col_num], dtype=np.int8)
    vert_AOD_cmds = np.zeros([row_num], dtype=np.int8)
    parallel_success_flag = True

    # Generate AOD commands for a given row and column number
    for move in move_seq:
        # Change the status of vertical AOD commands
        if move.from_row > move.to_row:
            if vert_AOD_cmds[move.from_row] == 0:
                vert_AOD_cmds[move.from_row] = 3
            elif vert_AOD_cmds[move.from_row] != 3:
                parallel_success_flag = False
                break
        elif move.from_row < move.to_row:
            if vert_AOD_cmds[move.from_row] == 0:
                vert_AOD_cmds[move.from_row] = 2
            elif vert_AOD_cmds[move.from_row] != 2:
                parallel_success_flag = False
                break
        else:
            if vert_AOD_cmds[move.from_row] == 0:
                vert_AOD_cmds[move.from_row] = 1
            elif vert_AOD_cmds[move.from_row] != 1:
                parallel_success_flag = False
                break

        # Change the status of horizontal AOD commands
        if move.from_col > move.to_col:
            if horiz_AOD_cmds[move.from_col] == 0:
                horiz_AOD_cmds[move.from_col] = 3
            elif horiz_AOD_cmds[move.from_col] != 3:
                parallel_success_flag = False
                break
        elif move.from_col < move.to_col:
            if horiz_AOD_cmds[move.from_col] == 0:
                horiz_AOD_cmds[move.from_col] = 2
            elif horiz_AOD_cmds[move.from_col] != 2:
                parallel_success_flag = False
                break
        else:
            if horiz_AOD_cmds[move.from_col] == 0:
                horiz_AOD_cmds[move.from_col] = 1
            elif horiz_AOD_cmds[move.from_col] != 1:
                parallel_success_flag = False
                break

    if parallel_success_flag and verify:
        move_list = get_move_list_from_AOD_cmds(horiz_AOD_cmds, vert_AOD_cmds)
        matrix_from_AOD = move_atoms_noiseless(copy.deepcopy(matrix), move_list)
        matrix_from_seq = move_atoms_noiseless(copy.deepcopy(matrix), move_seq)

        if not np.array_equal(matrix_from_AOD, matrix_from_seq):
            parallel_success_flag = False

    return horiz_AOD_cmds, vert_AOD_cmds, parallel_success_flag


## Functions for moving atoms ##


def move_atoms(
    init_matrix: NDArray,
    moves: list[Move],
    error_model: ErrorModel | None = None,
    params: PhysicalParams | None = None,
    look_for_flag: bool = False,
    error_modeling: bool = False,
) -> tuple[NDArray, list[list]]:
    """
    Apply a batch of intended moves to a standalone occupancy matrix.

    Why this exists
    ---------------
    This is a soon-to-be-deprecated, algorithm-facing implementation of `AtomArray.move_atoms`.
    Several algorithms (e.g. BCv2) maintain their *own* copy of the occupancy matrix and update
    it alongside the Move list. This helper provides that update while honoring Move failure
    annotations (`fail_event`, `fail_flag`) produced by the error pipeline.

    Unsigned dtype safety
    ---------------------
    Some algorithm code historically stored occupancy in unsigned dtypes (e.g. uint8). That is
    fine for representing counts, but intermediate arithmetic during move application can
    transiently require signed operations (e.g. decrementing, sanity checks) and may otherwise
    wrap around on underflow.

    Policy implemented here:
      1) If `init_matrix` is unsigned, cast a working copy to a signed integer dtype.
      2) Perform all bookkeeping and updates on the signed working copy.
      3) Before returning, assert the result is nonnegative, then cast back to the original
         unsigned dtype.

    Parameters
    ----------
    init_matrix
        Occupancy matrix for a single species. Expected shape is (rows, cols) and integer dtype.
    move_list
        Intended moves to attempt in this timestep.
    error_model
        Error model used to generate probabilistic failure flags for the move set.
    look_for_flag
        Whether to consider failure flags in addition to Move.fail_event (legacy compatibility).
    Returns
    -------
    matrix_out
        Updated occupancy matrix (same dtype as `init_matrix`).
    failed_moves
        List of indices of moves that failed.
    flags
        List of integer FailureFlag values for the failed moves.
    """
    if error_model is None:
        error_model = ZeroNoise()
    if params is None:
        params = PhysicalParams()
    matrix_out = copy.deepcopy(init_matrix)
    if np.max(init_matrix) > 1:
        raise Exception("Variable `init_matrix` cannot have values outside of {0,1}. ")

    # make sure `moves` is a list and not just a singular `Move` object
    try:
        moves[0]
    except TypeError:
        moves = [moves]

    if error_modeling:
        # evaluating moves from error model
        moves = error_model.get_move_errors(init_matrix, moves)

    if not isinstance(init_matrix, np.ndarray):
        raise TypeError("init_matrix must be a numpy array.")
    # if init_matrix.ndim != 3:
    #     raise ValueError(f"init_matrix must be 3D (rows, cols), got ndim={init_matrix.ndim}.")
    if not np.issubdtype(init_matrix.dtype, np.integer):
        raise TypeError(
            f"init_matrix must have an integer dtype, got {init_matrix.dtype}."
        )

    orig_dtype = init_matrix.dtype
    needs_recast: bool = bool(np.issubdtype(orig_dtype, np.unsignedinteger))

    # Signed working copy to prevent unsigned underflow/overflow during intermediate arithmetic.
    # int16 is sufficient for typical occupancy counts; keep int32 for wider unsigned inputs.
    if needs_recast:
        work_dtype = np.int32 if init_matrix.itemsize > 1 else np.int16
        matrix_out: NDArray[np.integer] = init_matrix.astype(work_dtype, copy=True)
    else:
        matrix_out = np.array(init_matrix, copy=True)

    if matrix_out.size > 0 and int(np.max(matrix_out)) > 1:
        raise ValueError("There is a site with more than one atom.")

    # prescreening moves to remove any intersecting tweezers
    matrix_out, duplicate_move_inds = _find_and_resolve_crossed_moves(moves, matrix_out)

    # applying moves on
    matrix_out, failed_moves, flags = _apply_moves(
        init_matrix, matrix_out, moves, duplicate_move_inds, look_for_flag=look_for_flag
    )

    # if there are multiple atoms in a trap, they repel each other
    if np.max(matrix_out) > 1:
        for i in range(len(matrix_out)):
            for j in range(len(matrix_out[0])):
                if matrix_out[i, j] > 1:
                    matrix_out[i, j] = 0

    # calculating the time it took the atoms to be moved
    max_distance = 0
    for move in moves:
        dist = (
            move.distance * params.spacing
            + (error_model.putdown_time + error_model.pickup_time) * params.AOD_speed
        )
        if dist > max_distance:
            max_distance = dist

        move_time = max_distance / params.AOD_speed

    # evaluating atom loss process from error model
    matrix_out, _ = error_model.get_atom_loss(matrix_out, move_time, n_species=1)

    if matrix_out.size > 0 and int(np.min(matrix_out)) < 0:
        raise ValueError(
            "move_atoms produced negative occupancy; refusing to cast to unsigned dtype."
        )

    if needs_recast:
        return matrix_out.astype(orig_dtype, copy=False), [failed_moves, flags]

    return matrix_out, [failed_moves, flags]


def _get_duplicate_vals_from_list(lis: list) -> list:
    return [k for k, v in Counter(lis).items() if v > 1]


def _find_and_resolve_crossed_moves(
    move_list: list, matrix_copy: np.ndarray
) -> tuple[NDArray, list]:
    """
    Identifies sets of moves where the AOD tweezers cross over each other (and destroy the atoms).
    NB: this ONLY works for moves where you only move by one column or one row.
    """
    # 1. getting midpoints of moves
    midpoints = []
    for move in move_list:
        midpoints.append((move.midx, move.midy))

    # 2. Finding duplicate midpoints
    duplicate_vals = _get_duplicate_vals_from_list(midpoints)

    # 3. Sorting duplicate entries into distinct sets
    crossed_move_sets = []
    duplicate_move_inds = []
    for _ in range(len(duplicate_vals)):
        crossed_move_sets.append([])
    if len(crossed_move_sets) > 0:
        for m_ind, move in enumerate(move_list):
            try:
                d_ind = duplicate_vals.index((move.midx, move.midy))
                crossed_move_sets[d_ind].append(m_ind)
                duplicate_move_inds.append(m_ind)
            except ValueError:
                pass
        # 4. iterature through the sets of overlapping moves
        for crossed_move_set in crossed_move_sets:
            # 4.1. check to see if there are atoms that would be moved
            for move_ind in crossed_move_set:
                move = move_list[move_ind]
                if matrix_copy[move.from_row][move.from_col] == 1:
                    # 4.2. if so, check whether the tweezer fails to pick up the atom
                    # if it picks up the atom, then the atom is ejected due to the collision with the other tweezer
                    if move.fail_flag != 1:
                        matrix_copy[move.from_row][move.from_col] = 0
                else:
                    move.fail_flag = 3  # meaning that there is no atom to move
    return matrix_copy, duplicate_move_inds


def _apply_moves(
    init_matrix: NDArray,
    matrix_out: NDArray,
    moves: list,
    duplicate_move_inds: list | None = None,
    look_for_flag: bool = False,
) -> tuple[NDArray, list, list]:
    """
    Applies moves to an array of atoms (represented by `matrix_out`).
    The function assumes that any moves which involve crossing tweezers
    have already been filtered out by `find_and_resolve_crossed_moves()`.

    NB: `init_matrix` is the initial array before crossed moves were resolved,
    and `matrix_out` is the array following resolution of crossed moves.
    """
    if duplicate_move_inds is None:
        duplicate_move_inds = []
    failed_moves = []
    flags = []

    def _resolve_indices(
        r: int,
        c: int,
        mat: NDArray,
    ) -> tuple[int, int, bool]:
        """Resolve source indices and detect legacy swapped (col,row) convention."""
        n_rows, n_cols = mat.shape[0], mat.shape[1]
        if 0 <= r < n_rows and 0 <= c < n_cols:
            return r, c, False
        if 0 <= c < n_rows and 0 <= r < n_cols:
            return c, r, True
        raise IndexError(
            f"Index out of bounds for both orientations: ({r},{c}) on shape {mat.shape}"
        )

    # evaluate and run each move
    for move_ind, move in enumerate(moves):
        if move_ind in duplicate_move_inds:
            failed_moves.append(move_ind)
            flags.append(move.fail_flag)
            continue

        try:
            from_r, from_c, swapped = _resolve_indices(
                move.from_row, move.from_col, init_matrix
            )
        except IndexError:
            failed_moves.append(move_ind)
            flags.append(move.fail_flag)
            continue

        # fail flag code for the move: SUCCESS[0], PICKUPFAIL[1], PUTDOWNFAIL[2], NOATOM[3]
        # move.fail_flag = random.choices([1-pickup_fail_rate-putdown_fail_rate, pickup_fail_rate, putdown_fail_rate])[0]

        # Classify the move as:
        #   a) legal (there is an atom in the pickup position and NO atom in the putdown position),
        #   b) illegal (there is an atom in the pickup pos and an atom in the putdown pos)
        #   c) eject (there is an atom in the pickup pos and the putdown pos is outside of the array)
        #   d) no atom to move (there is NO atom in the pickup pos)
        if int(np.sum(init_matrix[from_r, from_c], dtype=np.int64)) == 1:
            try:
                # Resolve putdown indices using the same orientation as pickup.
                to_r, to_c = (
                    (move.to_row, move.to_col)
                    if not swapped
                    else (move.to_col, move.to_row)
                )
                if (
                    to_c >= 0
                    and to_r >= 0
                    and int(np.sum(init_matrix[to_r, to_c], dtype=np.int64)) == 0
                ):
                    movetype = MoveType.LEGAL_MOVE
                elif (
                    to_c >= 0
                    and to_r >= 0
                    and int(np.sum(init_matrix[to_r, to_c], dtype=np.int64)) == 1
                ):
                    movetype = MoveType.ILLEGAL_MOVE
                elif to_c >= 0 and to_r >= 0:
                    raise Exception(
                        f"{int(init_matrix[to_r][to_c])} is not a valid matrix entry."
                    )
                else:
                    raise IndexError
            except IndexError:
                movetype = MoveType.EJECT_MOVE
        else:
            movetype = MoveType.NO_ATOM_TO_MOVE
            move.fail_flag = 3

        # If the move fails due to pickup/putdown stochastic failure, record it.
        if move.fail_flag != 0 and look_for_flag:
            failed_moves.append(move_ind)
            flags.append(move.fail_flag)
            if move.fail_flag == 2:  # PUTDOWNFAIL
                if matrix_out[from_r][from_c] == 0:
                    raise Exception(
                        f"Error occured in MoveType. There is NO atom at ({from_r}, {from_c})."
                    )
                matrix_out[from_r][from_c] -= 1
        elif movetype == MoveType.LEGAL_MOVE or movetype == MoveType.ILLEGAL_MOVE:
            to_r, to_c = (
                (move.to_row, move.to_col)
                if not swapped
                else (move.to_col, move.to_row)
            )
            if matrix_out[from_r][from_c] > 0:
                matrix_out[from_r][from_c] -= 1
                if 0 <= to_r < matrix_out.shape[0] and 0 <= to_c < matrix_out.shape[1]:
                    matrix_out[to_r][to_c] += 1
        elif movetype == MoveType.EJECT_MOVE:
            if matrix_out[from_r][from_c] == 0:
                raise Exception(
                    f"Error occured in MoveType assignment. There is NO atom at ({from_r}, {from_c})."
                )
            matrix_out[from_r][from_c] -= 1
    return matrix_out, failed_moves, flags


## refactored version for algorithms


def _normalize_single_species_matrix(
    matrix: IntArray,
) -> tuple[IntArray, bool]:
    """
    Normalize a single-species occupancy array to shape (rows, cols, 1).

    Parameters
    ----------
    matrix : IntArray
        Occupancy array with shape (rows, cols) or (rows, cols, 1).

    Returns
    -------
    tuple[IntArray, bool]
        Normalized 3D array and whether the input was originally 2D.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy array.")
    if not np.issubdtype(matrix.dtype, np.integer):
        raise TypeError(f"matrix must have an integer dtype, got {matrix.dtype}.")

    if matrix.ndim == 2:
        return matrix[:, :, None], True

    if matrix.ndim == 3 and matrix.shape[2] == 1:
        return matrix, False

    raise ValueError(
        "matrix must have shape (rows, cols) or (rows, cols, 1) for the "
        "single-species deterministic planning helper."
    )


def _restore_single_species_matrix(
    matrix_3d: IntArray,
    *,
    was_2d: bool,
    out_dtype: np.dtype,
) -> IntArray:
    """
    Restore a normalized 3D occupancy array to the caller's original shape.

    Parameters
    ----------
    matrix_3d : IntArray
        Normalized 3D occupancy array.
    was_2d : bool
        Whether the caller originally passed a 2D array.
    out_dtype : np.dtype
        Output dtype to restore.

    Returns
    -------
    IntArray
        Occupancy array with the original dimensionality.
    """
    if was_2d:
        og_dim_arr = matrix_3d[:, :, 0].astype(out_dtype, copy=False)
        return og_dim_arr
    og_dim_arr = matrix_3d.astype(out_dtype, copy=False)
    return og_dim_arr


def detect_destructive_aod_cmd_mask(
    arr: NDArray[np.integer],
) -> BoolArray:
    """
    Detect destructive adjacent AOD-tone patterns in a 1D command array.

    Parameters
    ----------
    arr : NDArray[np.integer]
        1D AOD command array with entries in {0, 1, 2, 3}.

    Returns
    -------
    BoolArray
        Boolean mask marking tone indices involved in destructive patterns.

    Notes
    -----
    The destructive adjacent patterns are:
      - [2, 1]
      - [1, 3]
      - [2, 3]
    """
    arr_int8: NDArray[np.int8] = np.asarray(arr, dtype=np.int8)
    if arr_int8.ndim != 1:
        raise ValueError(f"arr must be 1D, got ndim={arr_int8.ndim}.")

    n: int = int(arr_int8.shape[0])
    mask: BoolArray = np.zeros(n, dtype=np.bool_)

    if n < 2:
        return mask

    mask[:-1] |= (arr_int8[:-1] == 2) & (arr_int8[1:] == 1)
    mask[1:] |= (arr_int8[:-1] == 2) & (arr_int8[1:] == 1)

    mask[:-1] |= (arr_int8[:-1] == 1) & (arr_int8[1:] == 3)
    mask[1:] |= (arr_int8[:-1] == 1) & (arr_int8[1:] == 3)

    mask[:-1] |= (arr_int8[:-1] == 2) & (arr_int8[1:] == 3)
    mask[1:] |= (arr_int8[:-1] == 2) & (arr_int8[1:] == 3)

    return mask


def build_destructive_support_mask(
    h_cmds: NDArray[np.integer],
    v_cmds: NDArray[np.integer],
) -> BoolArray:
    """
    Build the 2D support mask destroyed by collided AOD tones.

    Parameters
    ----------
    horiz_cmds : NDArray[np.integer]
        Horizontal AOD command array indexed by column.
    vert_cmds : NDArray[np.integer]
        Vertical AOD command array indexed by row.

    Returns
    -------
    BoolArray
        Boolean mask of shape (n_rows, n_cols) marking sites whose tweezer
        support is destroyed by collided tones.
    """

    if h_cmds.ndim != 1 or v_cmds.ndim != 1:
        raise ValueError("horiz_cmds and vert_cmds must both be 1D.")

    h_active: BoolArray = h_cmds != 0  # shape (n_cols,)
    v_active: BoolArray = v_cmds != 0  # shape (n_rows,)

    h_collided: BoolArray = detect_destructive_aod_cmd_mask(h_cmds)  # cols
    v_collided: BoolArray = detect_destructive_aod_cmd_mask(v_cmds)  # rows

    # shape (n_rows, n_cols)
    killed_by_h: BoolArray = v_active[:, None] & h_collided[None, :]
    killed_by_v: BoolArray = v_collided[:, None] & h_active[None, :]

    return killed_by_h | killed_by_v


def find_destructive_support_mask_from_moves(
    matrix: IntArray,
    moves: list[Move],
) -> tuple[BoolArray, bool]:
    """
    Infer the destructive support mask from a simultaneous move batch.

    Parameters
    ----------
    matrix : IntArray
        Occupancy matrix used for shape normalization and AOD command inference.
    moves : list[Move]
        Proposed simultaneous move batch.

    Returns
    -------
    tuple[BoolArray, bool]
        Destructive support mask and the representability flag returned by the
        AOD command inference helper.
    """
    matrix_3d: IntArray
    _was_2d: bool
    matrix_3d, _was_2d = _normalize_single_species_matrix(matrix)
    matrix_2d: IntArray = matrix_3d[:, :, 0]

    horiz_cmds, vert_cmds, ok = get_AOD_cmds_from_move_list(
        matrix_2d,
        moves,
    )
    support_mask: BoolArray = build_destructive_support_mask(horiz_cmds, vert_cmds)
    return support_mask, bool(ok)


def apply_moves_noiseless(
    init_matrix: IntArray,
    moves: list[Move],
    *,
    destructive_support_mask: BoolArray | None = None,
) -> IntArray:
    """
    Apply a simultaneous move batch under deterministic planning semantics.

    Why this exists
    ---------------
    Algorithm generation only cares about the next occupancy state under
    noiseless execution. This helper strips away probabilistic error processes
    and metadata bookkeeping, while retaining deterministic destructive AOD-tone
    collisions.

    Deterministic semantics
    -----------------------
    1. Any atom supported by a destructive AOD tone is ejected before move
       application.
    2. Surviving moves are classified against the initial occupancy matrix:
       - if the source initially contains an atom and the destination is
         in-bounds, decrement the source and increment the destination;
       - if the source initially contains an atom and the destination is
         out-of-bounds, decrement the source;
       - if the source initially contains no atom, do nothing.
    3. The output shape matches the input shape.

    Parameters
    ----------
    init_matrix : IntArray
        Occupancy array with shape (rows, cols) or (rows, cols, 1).
    moves : list[Move]
        Intended simultaneous moves.
    destructive_support_mask : BoolArray | None, optional
        Precomputed mask of sites whose tweezer support is destroyed.

    Returns
    -------
    IntArray
        Updated occupancy array with the same shape and dtype as the input.
    """
    matrix_3d: IntArray
    was_2d: bool
    matrix_3d, was_2d = _normalize_single_species_matrix(init_matrix)
    orig_dtype: np.dtype = init_matrix.dtype

    if np.issubdtype(orig_dtype, np.unsignedinteger):
        work_dtype: np.dtype = np.int32 if init_matrix.dtype.itemsize > 1 else np.int16
        init_work: IntArray = matrix_3d.astype(work_dtype, copy=True)
    else:
        init_work = np.array(matrix_3d, copy=True)

    if init_work.size > 0 and int(np.max(init_work)) > 1:
        raise ValueError("init_matrix cannot contain values outside of {0,1}.")

    out_work: IntArray = init_work.copy()
    init_2d: IntArray = init_work[:, :, 0]
    out_2d: IntArray = out_work[:, :, 0]

    if destructive_support_mask is None:
        destructive_support_mask, _ = find_destructive_support_mask_from_moves(
            init_work,
            moves,
        )

    if destructive_support_mask.shape != init_2d.shape:
        raise ValueError("destructive_support_mask has incompatible shape.")

    out_2d[destructive_support_mask] = 0

    n_rows: int = int(init_2d.shape[0])
    n_cols: int = int(init_2d.shape[1])

    for move in moves:
        src_row: int = int(move.from_row)
        src_col: int = int(move.from_col)
        dst_row: int = int(move.to_row)
        dst_col: int = int(move.to_col)

        if int(init_2d[src_row, src_col]) != 1:
            continue

        if out_2d[src_row, src_col] <= 0:
            continue

        out_2d[src_row, src_col] -= 1

        dst_in_bounds: bool = 0 <= dst_row < n_rows and 0 <= dst_col < n_cols
        if dst_in_bounds:
            out_2d[dst_row, dst_col] += 1

    out_2d[out_2d > 1] = 0
    return _restore_single_species_matrix(
        out_work,
        was_2d=was_2d,
        out_dtype=orig_dtype,
    )


def move_atoms_noiseless(
    init_matrix: IntArray,
    moves: list[Move] | Move,
) -> IntArray:
    """
    Deterministic planning helper for simultaneous move application.

    Parameters
    ----------
    init_matrix : IntArray
        Occupancy array with shape (rows, cols) or (rows, cols, 1).
    moves : list[Move] | Move
        Intended simultaneous move batch.

    Returns
    -------
    IntArray
        Updated occupancy array with the same shape and dtype as the input.
    """
    try:
        moves[0]  # type: ignore[index]
    except TypeError:
        move_list: list[Move] = [moves]  # type: ignore[list-item]
    else:
        move_list = moves  # type: ignore[assignment]

    destructive_support_mask: BoolArray
    _ok: bool
    destructive_support_mask, _ok = find_destructive_support_mask_from_moves(
        init_matrix,
        move_list,
    )
    return apply_moves_noiseless(
        init_matrix,
        move_list,
        destructive_support_mask=destructive_support_mask,
    )


## fast version


def move_atoms_fast(
    init_matrix: NDArray,
    moves: list[Move],
    error_model: ErrorModel | None = None,
    params: PhysicalParams | None = None,
    look_for_flag: bool = False,
    error_modeling: bool = False,
) -> tuple[NDArray, list[list]]:
    """
    Apply a batch of intended moves to a standalone occupancy matrix.

    This is a conservative performance refactor of ``move_atoms``. It preserves
    the current legacy semantics used by the algorithm code, including the
    expectation that ``init_matrix`` is a 3D single-species occupancy array with
    shape ``(rows, cols, 1)``.

    Parameters
    ----------
    init_matrix
        Single-species occupancy matrix with integer dtype and shape
        ``(rows, cols, 1)``.
    moves
        Intended moves to attempt.
    error_model
        Error model used when ``error_modeling=True`` and for post-move atom
        loss evaluation.
    params
        Physical parameters used in move-time calculation.
    look_for_flag
        Whether to honor legacy ``fail_flag`` behavior during move application.
    error_modeling
        Whether to run move-level error modeling before application.

    Returns
    -------
    tuple[NDArray, list[list]]
        Updated occupancy matrix and ``[failed_moves, flags]``.
    """
    if error_model is None:
        error_model = ZeroNoise()
    if params is None:
        params = PhysicalParams()
    if not isinstance(init_matrix, np.ndarray):
        raise TypeError("init_matrix must be a numpy array.")
    if init_matrix.ndim != 3:
        raise ValueError(
            f"init_matrix must be 2D (rows, cols), got ndim={init_matrix.ndim}."
        )
    if not np.issubdtype(init_matrix.dtype, np.integer):
        raise TypeError(
            f"init_matrix must have an integer dtype, got {init_matrix.dtype}."
        )

    if init_matrix.size > 0 and int(np.max(init_matrix)) > 1:
        raise Exception("Variable `init_matrix` cannot have values outside of {0,1}. ")

    try:
        moves[0]
    except TypeError:
        moves = [moves]

    if error_modeling:
        moves = error_model.get_move_errors(init_matrix, moves)

    orig_dtype = init_matrix.dtype
    needs_recast: bool = bool(np.issubdtype(orig_dtype, np.unsignedinteger))

    if needs_recast:
        work_dtype = np.int32 if init_matrix.itemsize > 1 else np.int16
        matrix_out: NDArray[np.integer] = init_matrix.astype(work_dtype, copy=True)
    else:
        matrix_out = np.array(init_matrix, copy=True)

    if matrix_out.size > 0 and int(np.max(matrix_out)) > 1:
        raise ValueError("There is a site with more than one atom.")

    matrix_out, duplicate_move_inds = _find_and_resolve_crossed_moves_fast(
        moves,
        matrix_out,
    )

    matrix_out, failed_moves, flags = _apply_moves_fast(
        init_matrix,
        matrix_out,
        moves,
        duplicate_move_inds,
        look_for_flag=look_for_flag,
    )

    if matrix_out.size > 0:
        matrix_out[matrix_out > 1] = 0

    max_distance: float = 0.0
    pickup_putdown_distance: float = (
        error_model.putdown_time + error_model.pickup_time
    ) * params.AOD_speed
    for move in moves:
        dist: float = move.distance * params.spacing + pickup_putdown_distance
        if dist > max_distance:
            max_distance = dist
        move_time: float = max_distance / params.AOD_speed

    matrix_out, _ = error_model.get_atom_loss(matrix_out, move_time, n_species=1)

    if matrix_out.size > 0 and int(np.min(matrix_out)) < 0:
        raise ValueError(
            "move_atoms produced negative occupancy; refusing to cast to unsigned dtype."
        )

    if needs_recast:
        return matrix_out.astype(orig_dtype, copy=False), [failed_moves, flags]
    return matrix_out, [failed_moves, flags]


def _find_and_resolve_crossed_moves_fast(
    move_list: list[Move],
    matrix_copy: np.ndarray,
) -> tuple[NDArray, list[int]]:
    """
    Identify midpoint-crossing moves and apply the same legacy resolution logic
    as ``_find_and_resolve_crossed_moves``.

    Parameters
    ----------
    move_list
        Proposed move batch.
    matrix_copy
        Mutable occupancy matrix after dtype normalization.

    Returns
    -------
    tuple[NDArray, list[int]]
        Updated matrix and the move indices involved in crossed-tweezer groups.
    """
    midpoint_groups: dict[tuple[float, float], list[int]] = {}
    for move_ind, move in enumerate(move_list):
        midpoint: tuple[float, float] = (move.midx, move.midy)
        if midpoint in midpoint_groups:
            midpoint_groups[midpoint].append(move_ind)
        else:
            midpoint_groups[midpoint] = [move_ind]

    duplicate_move_inds: list[int] = []
    crossed_move_sets: list[list[int]] = [
        inds for inds in midpoint_groups.values() if len(inds) > 1
    ]

    for crossed_move_set in crossed_move_sets:
        duplicate_move_inds.extend(crossed_move_set)
        for move_ind in crossed_move_set:
            move: Move = move_list[move_ind]
            if matrix_copy[move.from_row][move.from_col][0] == 1:
                if move.fail_flag != 1:
                    matrix_copy[move.from_row][move.from_col][0] = 0
            else:
                move.fail_flag = 3

    return matrix_copy, duplicate_move_inds


def _apply_moves_fast(
    init_matrix: NDArray,
    matrix_out: NDArray,
    moves: list[Move],
    duplicate_move_inds: list[int] | None = None,
    look_for_flag: bool = False,
) -> tuple[NDArray, list[int], list[int]]:
    """
    Apply move outcomes to a single-species occupancy array.

    Parameters
    ----------
    init_matrix
        Initial occupancy array before crossed-move resolution.
    matrix_out
        Working occupancy array after crossed-move resolution.
    moves
        Move batch.
    duplicate_move_inds
        Indices of moves already consumed by midpoint-crossing resolution.
    look_for_flag
        Whether to honor legacy ``fail_flag`` behavior.

    Returns
    -------
    tuple[NDArray, list[int], list[int]]
        Updated matrix, failed move indices, and failure flags.
    """
    if duplicate_move_inds is None:
        duplicate_move_inds = []

    duplicate_move_ind_set: set[int] = set(duplicate_move_inds)
    failed_moves: list[int] = []
    flags: list[int] = []

    for move_ind, move in enumerate(moves):
        if move_ind in duplicate_move_ind_set:
            failed_moves.append(move_ind)
            flags.append(move.fail_flag)
            continue

        if int(init_matrix[move.from_row][move.from_col][0]) == 1:
            try:
                if (
                    int(init_matrix[move.to_row][move.to_col][0]) == 0
                    and move.to_col >= 0
                    and move.to_row >= 0
                ):
                    movetype = MoveType.LEGAL_MOVE
                elif (
                    int(init_matrix[move.to_row][move.to_col][0]) == 1
                    and move.to_col >= 0
                    and move.to_row >= 0
                ):
                    movetype = MoveType.ILLEGAL_MOVE
                elif move.to_col >= 0 and move.to_row >= 0:
                    raise Exception(
                        f"{int(init_matrix[move.to_row][move.to_col][0])} "
                        "is not a valid matrix entry."
                    )
                else:
                    raise IndexError
            except IndexError:
                movetype = MoveType.EJECT_MOVE
        else:
            movetype = MoveType.NO_ATOM_TO_MOVE
            move.fail_flag = 3

        if move.fail_flag != 0 and look_for_flag:
            failed_moves.append(move_ind)
            flags.append(move.fail_flag)
            if move.fail_flag == 2:
                if matrix_out[move.from_row][move.from_col][0] == 0:
                    raise Exception(
                        f"Error occured in MoveType. There is NO atom at "
                        f"({move.from_row}, {move.from_col})."
                    )
                matrix_out[move.from_row][move.from_col][0] -= 1
        elif movetype == MoveType.LEGAL_MOVE or movetype == MoveType.ILLEGAL_MOVE:
            if matrix_out[move.from_row][move.from_col][0] > 0:
                matrix_out[move.from_row][move.from_col][0] -= 1
                matrix_out[move.to_row][move.to_col][0] += 1
        elif movetype == MoveType.EJECT_MOVE:
            if matrix_out[move.from_row][move.from_col][0] == 0:
                raise Exception(
                    f"Error occured in MoveType assignment. There is NO atom at "
                    f"({move.from_row}, {move.from_col})."
                )
            matrix_out[move.from_row][move.from_col][0] -= 1

    return matrix_out, failed_moves, flags
