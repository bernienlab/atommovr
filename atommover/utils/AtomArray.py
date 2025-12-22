# Base object to represent atom arrays

import copy
import math
import random
from collections import Counter

import numpy as np

from atommover.utils.animation import dual_species_image, single_species_image
from atommover.utils.core import (
    ArrayGeometry,
    Configurations,
    PhysicalParams,
    generate_middle_fifty,
    random_loading,
)
from atommover.utils.customize import SPECIES1NAME, SPECIES2NAME
from atommover.utils.ErrorModel import ErrorModel
from atommover.utils.errormodels import ZeroNoise
from atommover.utils.Move import Move
from atommover.utils.move_utils import MoveType


class AtomArray:
    """
    Base object representing the state of the atom array.

    ## Parameters
    shape : list[int,int]
        the number of columns and rows in the atom array.
    n_species : int
        whether the atom array is single- or dual-species. must be in [1,2].
    params : PhysicalParams
        the physical parameters describing tweezer and array properties.
    error_model : ErrorModel (or child class)
        an error model object that describes the physical error process (e.g.
        imperfect tweezer transfer, error from background gas collisions).
    geom : ArrayGeometry
        the geometry of the array.

    ## Example usage
    ```
    n_cols, n_rows = 10,10
    n_species = 1
    error_model = UniformVacuumTweezerError()
    tweezer_array = AtomArray([n_cols, n_rows],n_species, error_model=error_model)
    ```
    """

    def __init__(
        self,
        shape: list = [10, 10],
        n_species: int = 1,
        params: PhysicalParams = PhysicalParams(),
        error_model: ErrorModel = ZeroNoise(),
        geom: ArrayGeometry = ArrayGeometry.RECTANGULAR,
    ):
        self.geom = geom
        if n_species in [1, 2] and type(n_species) == int:
            self.n_species = n_species
        else:
            raise ValueError(
                f"Invalid entry for parameter `n_species`: {n_species}. The simulator only supports single and dual species arrays. "
            )
        self.shape = shape
        self.params = params
        self.error_model = error_model

        self.matrix = np.zeros([self.shape[0], self.shape[1], self.n_species])
        self.target = np.zeros([self.shape[0], self.shape[1], self.n_species])
        self.target_Rb = np.zeros([self.shape[0], self.shape[1]])
        self.target_Cs = np.zeros([self.shape[0], self.shape[1]])

    def __setattr__(self, key, value):
        if key == "shape":
            self.matrix = np.zeros([value[0], value[1], self.n_species])
            self.target = np.zeros([value[0], value[1], self.n_species])
            self.target_Rb = np.zeros([value[0], value[1]])
            self.target_Cs = np.zeros([value[0], value[1]])
            # (Optional) run any custom logic here
        # Always delegate to superclass to avoid recursion
        super().__setattr__(key, value)

    def load_tweezers(self) -> None:
        """
        Simulates uniform stochastic loading for single- or dual-species atom arrays.
        Loading probability (default 60%) can be set with `AtomArray.params.loading_prob`
        """
        if self.n_species == 1:
            self.matrix[:, :, :] = random_loading(
                self.shape, self.params.loading_prob
            ).reshape(self.shape[0], self.shape[1], 1)
        if self.n_species == 2:
            dual_species_prob = 2 - 2 * math.sqrt(1 - self.params.loading_prob)
            self.matrix[:, :, 0] = random_loading(self.shape, dual_species_prob / 2)
            self.matrix[:, :, 1] = random_loading(self.shape, dual_species_prob / 2)

            # Randomly leave one atom if there are two atoms share the same (x,y) coordinate
            for i in range(len(self.matrix)):
                for j in range(len(self.matrix[0])):
                    if self.matrix[i][j][0] == 1 and self.matrix[i][j][1] == 1:
                        random_index = random.randint(0, 1)
                        self.matrix[i][j][random_index] = 0

        self.last_loaded_config = copy.deepcopy(self.matrix)

    def generate_target(
        self,
        pattern: Configurations = Configurations.CHECKERBOARD,
        middle_size: list = [],
        occupation_prob: float = 0.5,
    ) -> None:
        if self.n_species == 1:
            self._generate_single_species_target(
                pattern, middle_size=middle_size, occupation_prob=occupation_prob
            )
        elif self.n_species == 2:
            self._generate_dual_species_target(
                pattern, middle_size=middle_size, occupation_prob=occupation_prob
            )
        else:
            raise ValueError(
                f"Unrecognized entry '{self.n_species}' for parameter `n_species`. The simulator only supports single and dual species arrays."
            )

    def _generate_single_species_target(
        self,
        pattern: Configurations = Configurations.MIDDLE_FILL,
        middle_size: list = [],
        occupation_prob: float = 0.5,
    ) -> None:
        """
        A function for generating common target configurations,
        such as checkerboard, zebra stripes, and middle fill.
        """
        array = np.zeros([self.shape[0], self.shape[1]])

        if len(middle_size) == 0:
            middle_size = generate_middle_fifty(self.shape[0], occupation_prob)

        if pattern == Configurations.ZEBRA_HORIZONTAL:  # every other row
            for i in range(0, self.shape[0], 2):
                array[i, :] = 1
        elif pattern == Configurations.ZEBRA_VERTICAL:  # every other col
            for i in range(0, self.shape[1], 2):
                array[:, i] = 1
        elif pattern == Configurations.CHECKERBOARD:  # checkerboard
            array = np.indices(self.shape).sum(axis=0) % 2
        elif pattern == Configurations.MIDDLE_FILL:  # middle fill
            mrow = np.zeros([1, self.shape[1]])
            mrow[
                0,
                int(self.shape[1] / 2 - middle_size[1] / 2) : int(
                    self.shape[1] / 2 - middle_size[1] / 2
                )
                + middle_size[1],
            ] = 1
            for i in range(
                int(self.shape[0] / 2 - middle_size[0] / 2),
                int(self.shape[0] / 2 - middle_size[0] / 2) + middle_size[0],
            ):
                array[i, :] = mrow
        elif pattern == Configurations.Left_Sweep:
            for i in range(middle_size[0]):
                array[:, i] = 1
        elif pattern == Configurations.RANDOM:
            array = random_loading(
                self.shape, probability=self.params.target_occup_prob
            )
        self.target = array.reshape([self.shape[0], self.shape[1], 1])

    def _generate_dual_species_target(
        self,
        pattern: Configurations = Configurations.ZEBRA_HORIZONTAL,
        middle_size: list = [],
        occupation_prob: float = 0.5,
    ) -> None:
        """
        A function for generating common target configurations,
        such as checkerboard, zebra stripes, and middle fill,
        for dual-species.
        """
        self.target_Rb = np.zeros([self.shape[0], self.shape[1]])
        self.target_Cs = np.zeros([self.shape[0], self.shape[1]])

        if len(middle_size) == 0:
            middle_size = generate_middle_fifty(self.shape[0], occupation_prob)

        # Horizontal zebra stripes mixed species pattern
        if pattern == Configurations.ZEBRA_HORIZONTAL:
            for i in range(
                int(self.shape[0] / 2 - middle_size[0] / 2),
                int(self.shape[0] / 2 - middle_size[0] / 2) + middle_size[0],
            ):
                for j in range(
                    int(self.shape[1] / 2 - middle_size[1] / 2),
                    int(self.shape[1] / 2 - middle_size[1] / 2) + middle_size[1],
                ):
                    if i % 2 == 0:
                        self.target_Cs[i, j] = 1
                    else:
                        self.target_Rb[i, j] = 1

        # Vertical zebra stripes mixed species pattern
        if pattern == Configurations.ZEBRA_VERTICAL:
            for i in range(
                int(self.shape[0] / 2 - middle_size[0] / 2),
                int(self.shape[0] / 2 - middle_size[0] / 2) + middle_size[0],
            ):
                for j in range(
                    int(self.shape[1] / 2 - middle_size[1] / 2),
                    int(self.shape[1] / 2 - middle_size[1] / 2) + middle_size[1],
                ):
                    if j % 2 == 0:
                        self.target_Cs[i, j] = 1
                    else:
                        self.target_Rb[i, j] = 1

        if pattern == Configurations.CHECKERBOARD:
            for i in range(
                int(self.shape[0] / 2 - middle_size[0] / 2),
                int(self.shape[0] / 2 - middle_size[0] / 2) + middle_size[0],
            ):
                for j in range(
                    int(self.shape[1] / 2 - middle_size[1] / 2),
                    int(self.shape[1] / 2 - middle_size[1] / 2) + middle_size[1],
                ):
                    if (i + j) % 2 == 0:
                        self.target_Rb[i, j] = 1
                    else:
                        self.target_Cs[i, j] = 1

        if pattern == Configurations.SEPARATE:
            for i in range(
                int(self.shape[0] / 2 - middle_size[0] / 2),
                int(self.shape[0] / 2 - middle_size[0] / 2) + middle_size[0],
            ):
                for j in range(
                    int(self.shape[1] / 2 - middle_size[1] / 2),
                    int(self.shape[1] / 2 - middle_size[1] / 2) + middle_size[1],
                ):
                    if j < int(self.shape[1] / 2):
                        self.target_Cs[i, j] = 1
                    else:
                        self.target_Rb[i, j] = 1

        self.target = np.stack([self.target_Rb, self.target_Cs], axis=2)

    def move_atoms(self, move_list: list[Move]) -> tuple[list, float]:
        if np.max(self.matrix) > 1:
            raise Exception("Atom array cannot have values outside of {0,1}.")

        # evaluating moves from error model and assigning failure flags
        moves_w_flags = self.error_model.get_move_errors(
            state=self.matrix, moves=move_list
        )

        # prescreening moves to remove any intersecting tweezers
        moves_wo_crossing, duplicate_move_inds = self._find_and_resolve_crossed_moves(
            moves_w_flags
        )

        # applying moves on current state of array
        failed_moves, flags = self._apply_moves(moves_wo_crossing, duplicate_move_inds)

        # if there are multiple atoms in a trap, they repel each other
        if self.n_species == 1:
            if np.max(self.matrix) > 1:
                for i in range(len(self.matrix)):
                    for j in range(len(self.matrix[0])):
                        if self.matrix[i, j] > 1:
                            self.matrix[i, j] = 0
        elif self.n_species == 2:
            if np.max(self.matrix[:, :, 0] + self.matrix[:, :, 1]) > 1:
                for i in range(len(self.matrix)):
                    for j in range(len(self.matrix[0])):
                        if self.matrix[i, j, 0] + self.matrix[i, j, 1] > 1:
                            self.matrix[i, j, 0] = 0
                            self.matrix[i, j, 1] = 0

        # calculating the time it took the atoms to be moved
        max_distance = 0
        for move in moves_wo_crossing:
            dist = (
                move.distance * self.params.spacing
                + (self.error_model.putdown_time + self.error_model.pickup_time)
                * self.params.AOD_speed
            )
            if dist > max_distance:
                max_distance = dist

        move_time = max_distance / self.params.AOD_speed

        # applying error model to calculate atom loss during the time the move was applied
        self.matrix, _ = self.error_model.get_atom_loss(
            self.matrix, evolution_time=move_time, n_species=self.n_species
        )

        return [failed_moves, flags], move_time

    def _get_duplicate_vals_from_list(self, l):
        return [k for k, v in Counter(l).items() if v > 1]

    def _find_and_resolve_crossed_moves(self, move_list: list) -> tuple[list, list]:
        """
        This function looks for situations where two moving tweezer paths cross over one another.

        In this situation, it is assumed that any atoms being carried in the tweezers are lost.
        If any such tweezers are found, the atoms are removed, and move.failure_flag is changed to 4.
        """
        # 1. getting midpoints of moves
        midpoints = []

        for move in move_list:
            midpoints.append((move.midx, move.midy))

        # 2. Finding duplicate midpoints
        duplicate_vals = self._get_duplicate_vals_from_list(midpoints)

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
                    if np.sum(self.matrix[move.from_row, move.from_col, :]) == 1:
                        # 4.2. if so, check whether the tweezer fails to pickup the atom
                        if move.failure_flag != 1:
                            self.matrix[move.from_row][move.from_col][0] = 0
                            if self.n_species == 2:
                                self.matrix[move.from_row][move.from_col][1] = 0
                            move.failure_flag = 4
                            move_list[move_ind] = move

                    else:
                        move.failure_flag = 3  # no atom to move
        return move_list, duplicate_move_inds

    def _apply_moves(
        self, moves: list, duplicate_move_inds: list = []
    ) -> tuple[list, list]:
        """
        Applies moves to an array of atoms (represented by `matrix_out`).
        The function assumes that any moves which involve crossing tweezers
        have already been filtered out by `find_and_resolve_crossed_moves()`.

        ## Parameters
        moves : list
            list of 'Move' objects to implement in *parallel*.
        duplicate_move_inds : list
            output of `find_and_resolve_crossed_moves()`

        ## Returns
        failed_moves : list
            list of the indices of moves that failed. Indices correspond to
            the order of the moves as presented in `moves`.
        flags : list
            the corresponding failure flags of the failed moves (identifies
            the type of failure).
        """
        if self.n_species == 1:
            return self._apply_moves_single_species(moves, duplicate_move_inds)
        elif self.n_species == 2:
            return self._apply_moves_dual_species(moves, duplicate_move_inds)
        else:
            return [], []

    def _apply_moves_single_species(
        self, moves: list, duplicate_move_inds: list = []
    ) -> tuple[list, list]:
        """
        Applies moves to an array of atoms (represented by `matrix_out`).
        The function assumes that any moves which involve crossing tweezers
        have already been filtered out by `find_and_resolve_crossed_moves()`.
        """
        state_before_moves = copy.deepcopy(self.matrix)
        failed_moves = []
        flags = []
        # evaluate and run each move
        for move_ind, move in enumerate(moves):
            # check if the move crosses another move, and if so, it was already dealt with in `_find_and_resolve_crossed_moves()``
            if move_ind in duplicate_move_inds:
                failed_moves.append(move_ind)
                flags.append(move.failure_flag)
                move.movetype = MoveType.ILLEGAL_MOVE
            else:
                # fail flag code for the move: SUCCESS[0], PICKUPFAIL[1], PUTDOWNFAIL[2], NOATOM[3], CROSSED[4]

                # Classify the move as:
                #   a) legal (there is an atom in the pickup position and NO atom in the putdown position),
                #   b) illegal (there is an atom in the pickup pos and an atom in the putdown pos)
                #   c) eject (there is an atom in the pickup pos and the putdown pos is outside of the array)
                #   d) no atom to move (there is NO atom in the pickup pos)
                #   e) crossed (the moving tweezer intersects with another moving tweezer and the atom will be lost)

                # if there is an atom in the pickup pos
                if int(state_before_moves[move.from_row][move.from_col]) == 1:

                    if (
                        move.to_col > self.shape[1] - 1
                        or move.to_row > self.shape[0] - 1
                        or move.to_col < 0
                        or move.to_row < 0
                    ):
                        move.movetype = MoveType.EJECT_MOVE
                    else:
                        # if the putdown pos is vacant, the move is legal
                        if int(state_before_moves[move.to_row][move.to_col]) == 0:
                            move.movetype = MoveType.LEGAL_MOVE
                        # if the putdown pos is filled, the move is illegal/there will be a collision
                        elif int(state_before_moves[move.to_row][move.to_col]) == 1:
                            move.movetype = MoveType.ILLEGAL_MOVE
                        else:
                            raise Exception(
                                f"{int(self.matrix[move.to_row][move.to_col])} is not a valid matrix entry."
                            )
                else:  # if there is no atom in the pickup pos
                    move.movetype = MoveType.NO_ATOM_TO_MOVE
                    move.failure_flag = 3

                # if the move fails due to the atom not being picked up or put down correctly, make a note of this.
                if move.failure_flag != 0:
                    failed_moves.append(move_ind)
                    flags.append(move.failure_flag)
                    if move.failure_flag == 2:  # PUTDOWNFAIL, see above
                        if state_before_moves[move.from_row][move.from_col] == 0:
                            raise Exception(
                                f"Error occured in MoveType. There is NO atom at ({move.from_row}, {move.from_col})."
                            )
                        self.matrix[move.from_row][move.from_col] -= 1
                # otherwise, if the target site is on the grid, you can implement it.
                # NB: the double occupation of a site from a illegal/collision move will be detected later in `move_atoms`
                elif (
                    move.movetype == MoveType.LEGAL_MOVE
                    or move.movetype == MoveType.ILLEGAL_MOVE
                ):
                    if state_before_moves[move.from_row][move.from_col] > 0:
                        self.matrix[move.from_row][move.from_col] -= 1
                        self.matrix[move.to_row][move.to_col] += 1
                # elif the move is an ejection move or moves an atom into an occupied site, remove the atom(s)
                elif move.movetype == MoveType.EJECT_MOVE:
                    if state_before_moves[move.from_row][move.from_col] == 0:
                        raise Exception(
                            f"Error occured in MoveType assignment. There is NO atom at ({move.from_row}, {move.from_col})."
                        )
                    self.matrix[move.from_row][move.from_col] -= 1
        return failed_moves, flags

    def _apply_moves_dual_species(
        self, moves: list, duplicate_move_inds: list = []
    ) -> tuple[list, list]:
        """
        Applies moves on the dual species atom array.
        """
        failed_moves = []
        flags = []
        # evaluate and run each move
        for move_ind, move in enumerate(moves):
            if move_ind in duplicate_move_inds:
                failed_moves.append(move_ind)
                flags.append(move.failure_flag)
            else:
                # key for `Move` attribute `failure_flag`: SUCCESS[0], PICKUPFAIL[1], PUTDOWNFAIL[2], NOATOM[3], CROSSED[4]

                # Classify the move as:
                #   a) legal (there is an atom in the pickup position and NO atom in the putdown position),
                #   b) illegal/collision (there is an atom in the pickup pos and an atom in the putdown pos)
                #   c) eject (there is an atom in the pickup pos and the putdown pos is outside of the array)
                #   d) no atom to move (there is NO atom in the pickup pos)
                #   e) crossed (the moving tweezer intersects with another moving tweezer and the atom will be lost)

                # if there is an atom in the pickup pos
                if int(np.sum(self.matrix[move.from_row, move.from_col, :])) == 1:
                    try:
                        # check if there is NO atom in the putdown pos
                        if (
                            int(np.sum(self.matrix[move.to_row, move.to_col, :])) == 0
                            and move.to_col >= 0
                            and move.to_row >= 0
                        ):
                            move.movetype = MoveType.LEGAL_MOVE
                        # check if there is an atom in the putdown pos
                        elif (
                            int(np.sum(self.matrix[move.to_row, move.to_col, :])) == 1
                            and move.to_col >= 0
                            and move.to_row >= 0
                        ):
                            move.movetype = MoveType.ILLEGAL_MOVE
                        elif move.to_col >= 0 and move.to_row >= 0:
                            raise Exception(
                                f"{int(np.sum(self.matrix[move.to_row,move.to_col,:]))} is not a valid matrix entry."
                            )
                        else:
                            raise IndexError
                    except IndexError:
                        move.movetype = MoveType.EJECT_MOVE
                else:  # if there is no atom in the pickup pos
                    move.movetype = MoveType.NO_ATOM_TO_MOVE
                    move.failure_flag = 3

                # if the move fails due to the atom not being picked up or put down correctly, make a note of this.
                if move.failure_flag != 0:
                    failed_moves.append(move_ind)
                    flags.append(move.failure_flag)
                    if move.failure_flag == 2:  # PUTDOWNFAIL, see above
                        if (
                            int(np.sum(self.matrix[move.from_row, move.from_col, :]))
                            == 0
                        ):
                            raise Exception(
                                f"Error occured in MoveType. There is NO atom at ({move.from_row}, {move.to_row})."
                            )
                        self.matrix[move.from_row][move.from_col][0] = 0
                        self.matrix[move.from_row][move.from_col][1] = 0

                # elif the move is valid, implement it
                elif (
                    move.movetype == MoveType.LEGAL_MOVE
                    or move.movetype == MoveType.ILLEGAL_MOVE
                ):
                    # If there is a Rb atom, move it
                    if self.matrix[move.from_row][move.from_col][0] > 0:
                        self.matrix[move.from_row][move.from_col][0] -= 1
                        self.matrix[move.to_row][move.to_col][0] += 1

                    # elif there is a Cs atom, move it
                    elif self.matrix[move.from_row][move.from_col][1] > 0:
                        self.matrix[move.from_row][move.from_col][1] -= 1
                        self.matrix[move.to_row][move.to_col][1] += 1

                # elif the move is an ejection move, remove the atom(s)
                elif move.movetype == MoveType.EJECT_MOVE:
                    if np.sum(self.matrix[move.from_row, move.from_col, :]) == 0:
                        raise Exception(
                            f"Error occured in MoveType assignment. There is NO atom at ({move.from_row}, {move.to_row})."
                        )
                    # Eject Rb atom
                    if self.matrix[move.from_row][move.from_col][0] > 0:
                        self.matrix[move.from_row][move.from_col][0] -= 1
                    # Eject Cs atom
                    elif self.matrix[move.from_row][move.from_col][1] > 0:
                        self.matrix[move.from_row][move.from_col][1] -= 1
        return failed_moves, flags

    def image(
        self, move_list: list = [], plotted_species: str = "all", savename=""
    ) -> None:
        f"""
        Takes a snapshot of the atom array.
        ## Parameters
        move_list : list
            any moves that you want to plot
        plotted_species : str 
            the atomic species you want to plot. You can choose from ['{SPECIES1NAME}','{SPECIES2NAME}','all'].
        """
        # make sure `moves` is a list and not just a singular `Move` object
        if not np.array_equal(move_list, []):
            try:
                move_list[0]
            except TypeError:
                move_list = [move_list]

        if self.n_species == 1:
            single_species_image(self.matrix, move_list=move_list, savename=savename)
        elif self.n_species == 2:
            if type(self) != np.ndarray:
                plotted_arrays = self.matrix
            else:
                plotted_arrays = self

            if (
                plotted_species.lower() == "all"
                or plotted_species.lower() == SPECIES1NAME.lower()
                or plotted_species.lower() == SPECIES2NAME.lower()
            ):
                dual_species_image(
                    plotted_arrays, move_list=move_list, savename=savename
                )
            else:
                raise ValueError(
                    f"Invalid entry for parameter 'plotted_species': {plotted_species}. Please choose from ['{SPECIES1NAME}','{SPECIES2NAME}', 'all']."
                )

    def plot_target_config(self) -> None:
        if self.n_species == 1:
            single_species_image(self.target)
        elif self.n_species == 2:
            dual_species_image(self.target)

    def evaluate_moves(self, move_list: list) -> tuple[float, list]:
        # making reference time
        t_total = 0
        N_parallel_moves = 0
        N_non_parallel_moves = 0

        # iterating through moves and updating internal state matrix
        for _, move_set in enumerate(move_list):

            # performing the move
            _, move_time = self.move_atoms(move_set)
            N_parallel_moves += 1
            N_non_parallel_moves += len(move_set)

            # calculating the time to complete the move set in parallel
            t_total += move_time

        return float(t_total), [N_parallel_moves, N_non_parallel_moves]
