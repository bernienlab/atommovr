# Dual-species algorithms.

# FOR CONTRIBUTORS:
# - Please write your algorithm in a separate .py file
# - Once you have done that, please make an algorithm class with the following three functions:
#   1. `__repr__(self)` - this should return the name of your algorithm, to be used in plots.
#   2. `get_moves(self)` - given an AtomArray object, returns a list of Move() objects.
#   3. (optional) `__init__()` - if your algorithm needs to use arguments that cannot be specified in AtomArray
# - see the `Algorithm` base class for more details/instructions.

from atommovr.utils.AtomArray import AtomArray
from atommovr.algorithms.Algorithm_class import Algorithm
from atommovr.algorithms.source.inside_out import inside_out_algorithm
from atommovr.algorithms.source.naive_parallel_Hung import naive_par_Hung

###########################################
# Existing algorithms from the literature #
###########################################


###########################################
# New algorithms proposed in our work #
###########################################


class InsideOut(Algorithm):
    """
    Implements the InsideOut algorithm.
    """

    def __repr__(self):
        return "InsideOut"

    def get_moves(self, dual_sp_array: AtomArray, do_ejection: bool = False):
        return inside_out_algorithm(dual_sp_array, do_ejection=do_ejection)


class NaiveParHung(Algorithm):
    """
    Implements a naive extension of ParHungarian for dual-species arrays.
    """

    def __repr__(self):
        return "NaiveParHung"

    def get_moves(self, dual_sp_array: AtomArray, do_ejection: bool = False):
        return naive_par_Hung(dual_sp_array, do_ejection=do_ejection)
