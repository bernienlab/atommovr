# Parent class for all algorithms.
# For developers:
#   Feel free to use this as a template for new algorithms.
#   Each function below describes the requirements/what you need to put in it.

# Author: Nikhil Harle

import numpy as np
from atommovr.utils.AtomArray import AtomArray


class Algorithm:
    """
    Parent class for all algorithms.

    NB: The following functions are placeholders for illustrative purposes only and
    should be overwritten for your particular algorithm.

    If your algorithm can only prepare select target configurations, please list them here.

    e.g:

    Supported configurations: Middle Fill (see `atommovr.utils.core.Configurations`
    for a list of configurations).
    """

    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "Insert the name of your algorithm here. This is what will show up on your benchmarking plots."

    def get_moves(
        self, atom_array, do_ejection: bool = False
    ) -> tuple[AtomArray, list, bool]:
        """
        This is the main function for the algorithm.

        ## Parameters
        **atom_array** : AtomArray
            object containing the initial configuration `atom_array.matrix`
            and the target configuration `atom_array.target`.

        **do_ejection** : bool, optional (default = False)
            argument to run an ejection subroutine(see
            `atommovr.algorithms.source.ejection.py` for the protocol).

        any other (optional!) kwargs you see fit to include :)

        ## Returns
        **config** : AtomArray
            the final configuration after all moves have been applied
            (ideally, this should just be the target configuration)

        **move_set** : list[list[Move, Move...], list[Move], ...]
            contains all the moves to transform the initial configuration into the final
            configuration.
            each list inside `move_set` is a set of moves that will be done in parallel.
            If you're confused by lists inside of lists, consider the following example:
                `small_move_list = [Move1]`
                `small_move_list1 = [Move2, Move3]`
                `small_move_list2 = [Move4, Move5]`
                `move_set = [small_move_list, small_move_list1, small_move_list2]`
            When this is read by the framework, it will first execute Move1, then will execute
            Move2 and Move3 in parallel, then after that Move4 and Move5 will be executed in
            parallel.
        **success_flag** : bool
            simple sanity check. This should be set to True if the algorithm prepares the
            final configuration and `False` if it does not. This is helpful during benchmarking.
        """
        config = AtomArray(shape=atom_array.shape, n_species=atom_array.n_species)
        move_set = []
        success_flag = False

        # your code here #

        return config, move_set, success_flag

    # Utility function common to all algorithms
    @staticmethod
    def get_success_flag(
        state: np.ndarray,
        target: np.ndarray,
        do_ejection: bool = False,
        n_species: int = 1,
    ) -> bool:
        """
        Checks if the target configuration was prepared and returns a flag.

        If do_ejection = False, the function only checks the square area around
        the target configuration. If True, it checks the entire array.

        ## Parameters
        FINISH THIS.

        """
        success_flag = False
        if not do_ejection:  # if do_ejection == False:
            start_row, end_row, start_col, end_col = get_effective_target_grid(
                target, n_species
            )
        else:
            start_row, start_col = 0, 0
            end_row, end_col = np.shape(state)[:2]
            end_row -= 1
            end_col -= 1

        if np.shape(state) != np.shape(target):
            print(
                f"Mismatch in shapes {np.shape(state)} and {np.shape(target)}. Reshaping."
            )
            state = state.reshape(np.shape(target))

        if n_species == 1:
            if np.array_equal(
                state[start_row : end_row + 1, start_col : end_col + 1],
                target[start_row : end_row + 1, start_col : end_col + 1],
            ):
                success_flag = True
        elif n_species == 2:
            if np.array_equal(
                state[start_row : end_row + 1, start_col : end_col + 1, :],
                target[start_row : end_row + 1, start_col : end_col + 1, :],
            ):
                success_flag = True

        return success_flag


def get_effective_target_grid(target, n_species=1):
    try:
        n_rows, n_cols = target.shape
    except ValueError:
        n_rows, n_cols, _ = target.shape
    for row_ind in range(n_rows):
        if n_species == 1:
            row = target[row_ind, :]
        else:
            row = target[row_ind, :, :]
        if 1 in row:
            start_row = row_ind
            break
    for row_ind in range(n_rows):
        if n_species == 1:
            row1 = target[n_rows - 1 - row_ind, :]
        else:
            row1 = target[n_rows - 1 - row_ind, :, :]
        if 1 in row1:
            end_row = n_rows - 1 - row_ind
            break

    for col_ind in range(n_cols):
        if n_species == 1:
            col = target[:, col_ind]
        else:
            col = target[:, col_ind, :]
        if 1 in col:
            start_col = col_ind
            break

    for col_ind in range(n_cols):
        if n_species == 1:
            col1 = target[:, n_cols - 1 - col_ind]
        else:
            col1 = target[:, n_cols - 1 - col_ind, :]
        if 1 in col1:
            end_col = n_cols - 1 - col_ind
            break
    try:
        return start_row, end_row, start_col, end_col
    except UnboundLocalError as ule:
        raise ValueError(
            "No atoms in target configuration. Did you initialize a target configuration with AtomArray.generate_target()?"
        ) from ule
