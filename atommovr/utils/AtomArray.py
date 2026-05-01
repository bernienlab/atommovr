"""
Authors: Nikhil Harle, Bo-Yu Chen

Description: Core class representing the state of the atom array.
Other supplementary classes are integrated by being passed to this class as properties
(e.g. error models, physical parameters).
"""

import copy
import math
import random
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List

from atommovr.utils.core import (
    PhysicalParams,
    ArrayGeometry,
    Configurations,
    random_loading,
    generate_middle_fifty,
)
from atommovr.utils.animation import dual_species_image, single_species_image
from atommovr.utils.move_utils import (
    MoveType,
    MultiOccupancyFlag,
    get_AOD_cmds_from_move_list,
    get_move_list_from_AOD_cmds,
    alloc_event_mask,
)
from atommovr.utils.Move import Move
from atommovr.utils.aod_timing import (
    _detect_pickup_and_accel_masks,
    _detect_decel_and_putdown_masks,
    _has_colliding_tones,
    _find_colliding_tones,
    collision_eligibility_from_tones,
)
from atommovr.utils.failure_policy import (
    FailureBit,
    FailureEvent,
    FailureFlag,
    bit_value,
    suppress_inplace,
    resolve_primary_events,
)
from atommovr.utils.ErrorModel import ErrorModel
from atommovr.utils.errormodels import ZeroNoise
from atommovr.utils.customize import SPECIES1NAME, SPECIES2NAME


class AtomArray:
    """
    Represent the simulated state of an atom array and its associated targets.

    This object stores the occupation state of a tweezer array, along with the
    physical configuration needed to simulate transport and error processes.
    It supports both single-species and dual-species arrays and initializes the
    corresponding state and target matrices used throughout the simulator.

    Parameters
    ----------
    shape : list[int, int], optional
        Array shape given as ``[n_cols, n_rows]``.
    n_species : int, optional
        Number of atomic species represented in the array. Must be either
        ``1`` or ``2``.
    params : PhysicalParams, optional
        Physical parameters describing tweezer and array properties.
    error_model : ErrorModel, optional
        Error model governing stochastic or physical failure processes during
        simulation, such as transfer infidelity or background-gas loss.
    geom : ArrayGeometry, optional
        Geometry of the array.

    Attributes
    ----------
    geom : ArrayGeometry
        Geometry of the array.
    shape : list[int, int]
        Array shape given as ``[n_cols, n_rows]``.
    n_species : int
        Number of atomic species represented in the simulator.
    params : PhysicalParams
        Physical parameters used by the array model.
    error_model : ErrorModel
        Error model applied during array operations.
    matrix : NDArray[np.uint8]
        Occupation matrix of shape ``(n_cols, n_rows, n_species)`` describing
        the current atom configuration.
    target : NDArray[np.uint8]
        Target occupation matrix of shape ``(n_cols, n_rows, n_species)``.
    target_Rb : NDArray[np.uint8]
        Single-species target projection for Rb atoms.
    target_Cs : NDArray[np.uint8]
        Single-species target projection for Cs atoms.

    Examples
    --------
    >>> n_cols, n_rows = 10, 10
    >>> error_model = UniformVacuumTweezerError()
    >>> tweezer_array = AtomArray(
    ...     [n_cols, n_rows],
    ...     n_species=1,
    ...     error_model=error_model,
    ... )
    """

    def __init__(
        self,
        shape: list | None = None,
        n_species: int = 1,
        params: PhysicalParams | None = None,
        error_model: ErrorModel | None = None,
        geom: ArrayGeometry = ArrayGeometry.RECTANGULAR,
    ):
        self.geom = geom
        if params is None:
            params = PhysicalParams()
        if error_model is None:
            error_model = ZeroNoise()
        if shape is None:
            shape = [10, 10]
        super().__setattr__("shape", shape)
        if n_species in [1, 2] and isinstance(n_species, int):
            self.n_species = n_species
        else:
            raise ValueError(
                f"Invalid entry for parameter `n_species`: {n_species}. The simulator only supports single and dual species arrays."
            )
        self.params = params
        self.error_model = error_model

        self.matrix = np.zeros(
            [self.shape[0], self.shape[1], self.n_species], dtype=np.uint8
        )
        self.target = np.zeros(
            [self.shape[0], self.shape[1], self.n_species], dtype=np.uint8
        )
        self.target_Rb = np.zeros([self.shape[0], self.shape[1]], dtype=np.uint8)
        self.target_Cs = np.zeros([self.shape[0], self.shape[1]], dtype=np.uint8)

    def __setattr__(self, key, value):
        if key == "shape":
            self.matrix = np.zeros([value[0], value[1], self.n_species], dtype=np.uint8)
            self.target = np.zeros([value[0], value[1], self.n_species], dtype=np.uint8)
            self.target_Rb = np.zeros([value[0], value[1]], dtype=np.uint8)
            self.target_Cs = np.zeros([value[0], value[1]], dtype=np.uint8)
            # (Optional) run any custom logic here
        # Always delegate to superclass to avoid recursion
        super().__setattr__(key, value)

    def load_tweezers(self) -> None:  # TODO: rewrite this to speed things up.
        """
        Populate the array with a stochastic initial loading configuration.

        This method samples a fresh occupation pattern for the current array shape
        using the loading probability stored in ``self.params.loading_prob``. For single-species
        arrays, each site is loaded independently. For dual-species arrays, each
        species is loaded independently and any site that ends up doubly occupied is
        reduced to a single randomly chosen species so that the post-loading state
        satisfies the single-atom-per-site constraint.

        The sampled configuration is written in place to ``self.matrix`` and copied
        to ``self.last_loaded_config`` for later reference.

        Returns
        -------
        None
            This method updates internal state in place.
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

        self.last_loaded_config = copy.deepcopy(self.matrix)  # can just use .copy()

    def generate_target(
        self,
        pattern: Configurations = Configurations.CHECKERBOARD,
        middle_size: list | None = None,
        occupation_prob: float = 0.5,
    ):
        """
        Generate a target occupation pattern for the current array.

        This method dispatches to the single-species or dual-species target builder
        depending on ``self.n_species``. It is the public entry point for creating
        common target configurations used by rearrangement algorithms and simulator
        diagnostics.

        Parameters
        ----------
        pattern : Configurations, optional
            Target pattern to generate.
        middle_size : list | None, optional
            Size specification for centered target patterns. When ``None``, the
            target builder chooses a default centered region based on the array size
            and occupation probability.
        occupation_prob : float, optional
            Filling fraction used when constructing default centered regions.

        Returns
        -------
        None
            This method updates ``self.target`` in place.

        Raises
        ------
        ValueError
            If ``self.n_species`` is not supported by the simulator.
        """
        if middle_size is None:
            middle_size = []
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

    def image(
        self,
        move_list: List[Move] | None = None,
        plotted_species: str = "all",
        savename: str = "",
    ) -> None:
        """
        Plot a snapshot of the current atom-array state.

        This method visualizes the current occupation matrix and can optionally
        overlay a set of moves. For dual-species arrays, the plot can show both
        species together or restrict the display to a single species-specific color
        scheme.

        Parameters
        ----------
        move_list : list[Move] | None, optional
            Moves to overlay on the plotted array. A single ``Move`` is promoted to
            a one-element list.
        plotted_species : str, optional
            Species selection for dual-species plots. Valid choices are
            ``'all'``, ``SPECIES1NAME``, and ``SPECIES2NAME``.
        savename : str, optional
            Output filename. If empty, the image is displayed without saving.

        Returns
        -------
        None
            This method produces a plot as a side effect.

        Raises
        ------
        ValueError
            If ``plotted_species`` is not one of the supported values for a
            dual-species plot.
        """
        # make sure `moves` is a list and not just a singular `Move` object
        if move_list is not None:
            try:
                move_list[0]
            except TypeError:
                move_list = [move_list]

        if self.n_species == 1:
            single_species_image(self.matrix, move_list=move_list, savename=savename)
        elif self.n_species == 2:
            if not isinstance(self, np.ndarray):
                plotted_arrays = self.matrix
            else:
                plotted_arrays = self

            if plotted_species.lower() == "all":
                dual_species_image(
                    plotted_arrays, move_list=move_list, savename=savename
                )
            elif plotted_species.lower() == SPECIES1NAME.lower():
                dual_species_image(
                    plotted_arrays,
                    color_scheme="blue",
                    move_list=move_list,
                    savename=savename,
                )
            elif plotted_species.lower() == SPECIES2NAME.lower():
                dual_species_image(
                    plotted_arrays,
                    color_scheme="yellow",
                    move_list=move_list,
                    savename=savename,
                )
            else:
                raise ValueError(
                    f"Invalid entry for parameter 'plotted_species': {plotted_species}. Please choose from ['{SPECIES1NAME}','{SPECIES2NAME}', 'all']."
                )

    def plot_target_config(self) -> None:
        """
        Plot the current target configuration.

        This is a convenience visualization method for inspecting the presently
        stored target occupation pattern. It chooses the appropriate plotting
        routine based on whether the array is single-species or dual-species.

        Returns
        -------
        None
            This method produces a plot as a side effect.
        """
        if self.n_species == 1:
            single_species_image(self.target)
        elif self.n_species == 2:
            dual_species_image(self.target)

    def get_target(self) -> NDArray[np.uint8]:
        """Return a copy of the target occupancy matrix.

        This compatibility shim preserves the legacy getter-style API used by
        older tests and downstream scripts.
        """
        return np.array(self.target, copy=True)

    def evaluate_moves(self, move_set: List) -> Tuple[float, List[int]]:
        """
        Execute a sequence of parallel move rounds and accumulate timing statistics.

        This method applies each round in ``move_set`` through ``move_atoms()``,
        updating the internal array state as it goes. It returns the total
        execution time together with simple bookkeeping counts for the number of
        parallel rounds and the total number of individual moves.

        Parameters
        ----------
        move_set : list
            Sequence of parallel move rounds, where each element is a list of
            ``Move`` objects to be executed together.

        Returns
        -------
        float
            Total simulated time required to execute the full move set.
        list[int]
            Two-element list ``[N_parallel_moves, N_non_parallel_moves]`` giving
            the number of parallel rounds and the total number of individual moves.
        """
        # making reference time
        t_total = 0
        N_parallel_moves = 0
        N_non_parallel_moves = 0

        # iterating through moves and updating internal state matrix
        for round_ind, move_list in enumerate(move_set):

            # performing the move
            if round_ind == 0:
                prev_move_list = []
            else:
                prev_move_list = move_set[round_ind - 1]
            try:
                next_move_list = move_set[round_ind + 1]
            except IndexError:
                next_move_list = []

            [_, _], move_time = self.move_atoms(
                move_list, prev_move_list, next_move_list
            )
            N_parallel_moves += 1
            N_non_parallel_moves += len(move_list)

            # calculating the time to complete the move set in parallel
            t_total += move_time

        return float(t_total), [N_parallel_moves, N_non_parallel_moves]

    def move_atoms(
        self,
        move_list: List[Move],
        prev_move_list: List[Move] | None = None,  # NEW for long moves upgrade
        next_move_list: List[Move] | None = None,  # NEW for long moves upgrade
    ) -> Tuple[List, float]:
        """
        Apply one parallel round of atom transport with failure sampling and timing.

        This method is the main transport step in the simulator. It validates that
        the proposed move round is physically parallelizable, constructs per-move
        eligibility masks for collision and timing-dependent processes, samples
        failures through the configured error model, applies the resulting move
        outcomes to ``self.matrix``, enforces post-apply occupancy constraints, and
        finally accounts for vacuum loss over the total move duration.

        The previous and next move lists are used to determine pickup, acceleration,
        deceleration, and putdown eligibility from AOD timing structure.

        Parameters
        ----------
        move_list : list[Move]
            Parallel move round to apply.
        prev_move_list : list[Move] | None, optional
            Previous parallel move round. Used to infer pickup and acceleration
            operations at the start of the current round.
        next_move_list : list[Move] | None, optional
            Next parallel move round. Used to infer deceleration and putdown
            operations at the end of the current round.

        Returns
        -------
        list
            Two-element list ``[failed_moves, flags]`` containing failed move
            indices (or move bookkeeping returned by ``_apply_moves``) and the
            corresponding failure flags accumulated during the round.
        float
            Total simulated time for the current move round.

        Raises
        ------
        ValueError
            If the occupation matrix is unphysical, if the move list is not
            parallelizable, or if the move list is incomplete relative to the
            active AOD tone intersections.
        """

        if prev_move_list is None:
            prev_move_list = []
        if next_move_list is None:
            next_move_list = []

        move_time = 0
        # 0. Sanity checks
        # 0.1 verify the matrix represents a physical state
        if np.max(self.matrix) > 1:
            raise ValueError("Atom array cannot have values outside of {0,1}.")
        if np.min(self.matrix) < 0:
            raise ValueError("Atom array cannot have negative occupancy values.")

        n_moves = len(move_list)
        if n_moves == 0:
            return [[], []], 0.0
        # 0.3 make sure the moves are physically parallelizable (for all three move lists)
        curr_horiz, curr_vert, curr_success = get_AOD_cmds_from_move_list(
            self.matrix, move_list
        )
        if not curr_success:
            raise ValueError(f"Non-parallelizable moves in move_list: {move_list}.")

        n_active_rows = int(np.count_nonzero(curr_horiz))
        n_active_cols = int(np.count_nonzero(curr_vert))
        expected_n_moves = n_active_rows * n_active_cols
        if n_moves != expected_n_moves:
            # quick and dirty patch
            h_cmds, v_cmds, success = get_AOD_cmds_from_move_list(
                self.matrix.copy(), move_list
            )
            move_list = get_move_list_from_AOD_cmds(h_cmds, v_cmds)
            n_moves = len(move_list)
            if not success:
                raise ValueError(
                    "Move list is not complete: active AOD tones define "
                    f"{n_active_rows} x {n_active_cols} = {expected_n_moves} "
                    f"tweezer intersections, but got {n_moves} moves."
                )

        if len(next_move_list) > 0:
            next_horiz, next_vert, next_success = get_AOD_cmds_from_move_list(
                self.matrix, next_move_list
            )
            if not next_success:
                raise ValueError(
                    f"Non-parallelizable moves in next_move_list: {next_move_list}."
                )
        else:
            next_horiz, next_vert = np.zeros(len(curr_horiz), dtype=np.int8), np.zeros(
                len(curr_vert), dtype=np.int8
            )

        if len(prev_move_list) > 0:
            prev_horiz, prev_vert, prev_success = get_AOD_cmds_from_move_list(
                self.matrix, prev_move_list
            )
            if not prev_success:
                raise ValueError(
                    f"Non-parallelizable moves in prev_move_list: {prev_move_list}."
                )
        else:
            prev_horiz, prev_vert = np.zeros(len(curr_horiz), dtype=np.int8), np.zeros(
                len(curr_vert), dtype=np.int8
            )

        # --- Building collision masks at the tone level ---
        has_colliding_tones = _has_colliding_tones(curr_vert, curr_horiz)
        if has_colliding_tones:
            (
                v_collision_inevitable,
                v_collision_avoidable,
                h_collision_inevitable,
                h_collision_avoidable,
            ) = _find_colliding_tones(curr_vert, curr_horiz)

        # --- Allocate per-move event mask (aligned 1:1 with move_list order) ---
        event_mask = alloc_event_mask(n_moves)

        # Vectorize move sources for mask-building
        source_rows = np.asarray([m.from_row for m in move_list], dtype=np.int_)
        source_cols = np.asarray([m.from_col for m in move_list], dtype=np.int_)

        # -------------------------------------------------------------------------
        # 0) Deterministic "NO_ATOM" tagging
        # -------------------------------------------------------------------------
        has_atom = np.sum(self.matrix[source_rows, source_cols, :], axis=-1) > 0
        no_atom_eligible = ~has_atom
        if no_atom_eligible.any():
            event_mask[no_atom_eligible] |= bit_value(FailureBit.NO_ATOM)

        # -------------------------------------------------------------------------
        # 1) Tagging moves with collisions
        #    NB: This replaces `_find_and_resolve_crossed_moves` (deprecated)
        # -------------------------------------------------------------------------
        if has_colliding_tones:
            eligible_collision_inevitable, eligible_collision_avoidable = (
                collision_eligibility_from_tones(
                    move_set=move_list,
                    v_collision_inevitable=v_collision_inevitable,
                    v_collision_avoidable=v_collision_avoidable,
                    h_collision_inevitable=h_collision_inevitable,
                    h_collision_avoidable=h_collision_avoidable,
                    source_rows=source_rows,
                    source_cols=source_cols,
                )
            )

            if eligible_collision_inevitable.any():
                self.error_model.apply_inevitable_collision_mask(
                    event_mask, eligible_collision_inevitable
                )

            if eligible_collision_avoidable.any():
                self.error_model.apply_avoidable_collision_mask(
                    event_mask, eligible_collision_avoidable
                )

        # -------------------------------------------------------------------------
        # 2) Pickup / Accel eligibility from AOD timing analysis
        # -------------------------------------------------------------------------
        eligible_pickup, eligible_accel = _detect_pickup_and_accel_masks(
            prev_horiz,
            prev_vert,
            curr_horiz,
            curr_vert,
            move_list,
            source_cols,
            source_rows,
        )

        # Time accounting + stochastic sampling into event_mask
        if eligible_pickup.any():
            move_time += self.error_model.pickup_time
            self.error_model.apply_pickup_errors_mask(event_mask, eligible_pickup)

        if eligible_accel.any():
            move_time += self.error_model.accel_time
            self.error_model.apply_accel_errors_mask(event_mask, eligible_accel)

        # -------------------------------------------------------------------------
        # 3) Decel / Putdown eligibility from AOD timing analysis
        # -------------------------------------------------------------------------
        eligible_decel, eligible_putdown = _detect_decel_and_putdown_masks(
            curr_horiz,
            curr_vert,
            next_horiz,
            next_vert,
            move_list,
            source_cols,
            source_rows,
        )

        if eligible_decel.any():
            move_time += self.error_model.decel_time
            self.error_model.apply_decel_errors_mask(event_mask, eligible_decel)

        if eligible_putdown.any():
            move_time += self.error_model.putdown_time
            self.error_model.apply_putdown_errors_mask(event_mask, eligible_putdown)

        # -------------------------------------------------------------------------
        # 4) Apply suppression policy + resolve to primary FailureEvent per move
        # -------------------------------------------------------------------------
        suppress_inplace(event_mask)
        primary_events = resolve_primary_events(event_mask)  # int FailureEvent codes

        # -------------------------------------------------------------------------
        # 5) Write primary event back onto Move objects (updates `fail_flag` internally)
        # -------------------------------------------------------------------------
        for i, move in enumerate(move_list):
            move.set_failure_event(int(primary_events[i]))

        # -------------------------------------------------------------------------
        # 6) Apply moves (with `fail_flag` and `fail_event` attributes)
        # -------------------------------------------------------------------------
        failed_moves, flags = self._apply_moves(move_list)

        # -------------------------------------------------------------------------
        # 7) Eject atoms from multi-occupied tweezers
        # -------------------------------------------------------------------------
        multi_occupancy_flags = self._eject_dual_occupied_sites_inplace(
            move_list, source_rows, source_cols
        )
        if len(multi_occupancy_flags) > 0:
            flags.extend(multi_occupancy_flags)

        if np.max(self.matrix) > 1:
            raise ValueError("Atom array cannot have values outside of {0,1}.")
        if np.min(self.matrix) < 0:
            raise ValueError("Atom array cannot have negative occupancy values.")

        # -------------------------------------------------------------------------
        # 8) Find maximum move time and sample from vacuum loss probability dist.
        # -------------------------------------------------------------------------
        max_distance = 0
        for move in move_list:  # moves_wo_crossing
            dist = move.distance * self.params.spacing
            if dist > max_distance:
                max_distance = dist
        move_time += max_distance / self.params.AOD_speed
        self.matrix, loss_flag = self.error_model.get_atom_loss(
            self.matrix, evolution_time=move_time, n_species=self.n_species
        )
        if loss_flag != 0:
            flags.append(loss_flag)

        return [failed_moves, flags], move_time

    def _eject_dual_occupied_sites_inplace(
        self,
        move_list: List[Move],
        source_rows: NDArray,
        source_cols: NDArray,
    ) -> List[MultiOccupancyFlag]:
        """
        Enforce the “≤ 1 atom per optical tweezer” invariant after modifying `matrix`
        from a list of `Move` objects.

        In this package, `move_atoms()` first samples and records single-move outcomes
        (via bitmasks → suppression → primary events), then `_apply_moves(...)` mutates
        the optical-tweezer occupancy accordingly. After that mutation, we enforce a
        global physical invariant: a tweezer is not allowed to end a timestep with
        more than one atom occupying it.

        This helper exists as a post-apply cleanup + bookkeeping step:
        - Eject (zero) any tweezers that ended up multi-occupied.
        - Tag the subset of moves that plausibly contributed to those multi-occupied
          tweezers, so users can count/diagnose how often post-apply cleanup occurs.

        Parameters
        ----------
        move_list
            Moves that have already had their `fail_event` / `fail_flag` written back
            (and have already been applied to `matrix` via `_apply_moves(...)`).
        source_rows, source_cols
            Vectorized sources for `move_list` (already computed upstream in
            `move_atoms()` to build eligibility masks).

        Returns
        -------
        list[MultiOccupancyFlag]
            Unordered list of flags that arose during this timestep from multi-occupancy
            cleanup. This list is sparse: it contains only flags for moves that were
            tagged (no “no error” placeholders).
        """
        flags: list[MultiOccupancyFlag] = []
        if len(move_list) == 0:
            return flags

        # --- 1) Detect post-apply multi-occupancy (matrix is assumed (H, W, n_species)) ---
        if self.n_species == 1:
            multi_occ_mask = self.matrix[:, :, 0] > np.uint8(1)
        elif self.n_species == 2:
            multi_occ_mask = (self.matrix[:, :, 0] + self.matrix[:, :, 1]) > np.uint8(1)
        else:
            raise ValueError(f"Unexpected n_species={self.n_species}. Expected 1 or 2.")

        if not np.any(multi_occ_mask):
            return flags

        # Eject any multi-occupied tweezers (clear all species channels at those sites).
        self.matrix[multi_occ_mask, :] = np.uint8(0)

        # --- 2) Identify which moves to tag ---
        # We tag:
        #   (a) any move whose destination is a multi-occupied tweezer
        #   (b) any move whose source is a multi-occupied tweezer AND it plausibly left
        #       an atom behind (failed-but-not-lost, excluding NO_ATOM and explicit EJECT)
        to_rows = np.asarray([m.to_row for m in move_list], dtype=np.int_)
        to_cols = np.asarray([m.to_col for m in move_list], dtype=np.int_)

        h, w = multi_occ_mask.shape
        coll_rows, coll_cols = np.where(multi_occ_mask)
        coll_site_ids = coll_rows.astype(np.int64) * np.int64(w) + coll_cols.astype(
            np.int64
        )

        to_site_ids = to_rows.astype(np.int64) * np.int64(w) + to_cols.astype(np.int64)
        from_site_ids = source_rows.astype(np.int64) * np.int64(w) + source_cols.astype(
            np.int64
        )

        hits_dest = np.isin(to_site_ids, coll_site_ids)

        # Conservative “left atom behind” check using move-level semantics already present.
        is_success = np.asarray([m.is_successful() for m in move_list], dtype=bool)
        is_loss = np.asarray([m.atom_was_lost() for m in move_list], dtype=bool)
        fail_flags = np.asarray([int(m.fail_flag) for m in move_list], dtype=np.int32)
        movetypes = np.asarray([int(m.movetype) for m in move_list], dtype=np.int32)

        left_atom_behind = (
            (~is_success)
            & (~is_loss)
            & (fail_flags != int(FailureFlag.NO_ATOM))
            & (movetypes != int(MoveType.EJECT_MOVE))
        )
        hits_source_and_left_atom = left_atom_behind & np.isin(
            from_site_ids, coll_site_ids
        )

        tagged = hits_dest | hits_source_and_left_atom
        idx = np.where(tagged)[0]

        for i in idx.tolist():
            move_list[i].multi_occupancy_flag = MultiOccupancyFlag.MULTI_ATOM_OCCUPANCY
            flags.append(MultiOccupancyFlag.MULTI_ATOM_OCCUPANCY)

        return flags

    def _generate_single_species_target(
        self,
        pattern: Configurations = Configurations.MIDDLE_FILL,
        middle_size: list | None = None,
        occupation_prob: float = 0.5,
    ) -> None:
        """
        Build a standard single-species target configuration.

        This helper constructs common target patterns used in rearrangement tasks,
        such as zebra stripes, checkerboards, centered fills, and random targets.
        The resulting pattern is written to ``self.target`` with a singleton
        species axis so that it matches the simulator's state representation.

        Parameters
        ----------
        pattern : Configurations, optional
            Target pattern to generate.
        middle_size : list, optional
            Size of the centered target region for patterns that use a middle
            window. If omitted or empty, a default centered size is generated from
            the array size and occupation probability.
        occupation_prob : float, optional
            Filling fraction used when selecting a default centered region.

        Returns
        -------
        None
            This method updates ``self.target`` in place.
        """
        array = np.zeros([self.shape[0], self.shape[1]], dtype=np.uint8)

        if middle_size is None or len(middle_size) == 0:
            middle_size = generate_middle_fifty(self.shape[0], occupation_prob)

        if pattern == Configurations.ZEBRA_HORIZONTAL:  # every other row
            for i in range(0, self.shape[0], 2):
                array[i, :] = np.uint8(1)
        elif pattern == Configurations.ZEBRA_VERTICAL:  # every other col
            for i in range(0, self.shape[1], 2):
                array[:, i] = np.uint8(1)
        elif pattern == Configurations.CHECKERBOARD:  # checkerboard
            array = (
                np.indices(self.shape, dtype=np.uint8).sum(axis=0, dtype=np.uint8) % 2
            )
        elif pattern == Configurations.MIDDLE_FILL:  # middle fill
            mrow = np.zeros([1, self.shape[1]], dtype=np.uint8)
            mrow[
                0,
                int(self.shape[1] / 2 - middle_size[1] / 2) : int(
                    self.shape[1] / 2 - middle_size[1] / 2
                )
                + middle_size[1],
            ] = np.uint8(1)
            for i in range(
                int(self.shape[0] / 2 - middle_size[0] / 2),
                int(self.shape[0] / 2 - middle_size[0] / 2) + middle_size[0],
            ):
                array[i, :] = mrow
        elif pattern == Configurations.Left_Sweep:
            for i in range(middle_size[0]):
                array[:, i] = np.uint8(1)
        elif pattern == Configurations.RANDOM:
            array = random_loading(
                self.shape, probability=self.params.target_occup_prob
            )
        self.target = array.reshape([self.shape[0], self.shape[1], 1])

    def _generate_dual_species_target(
        self,
        pattern: Configurations = Configurations.ZEBRA_HORIZONTAL,
        middle_size: list | None = None,
        occupation_prob: float = 0.5,
    ) -> None:
        """
        Build a standard dual-species target configuration.

        This helper constructs mixed-species target layouts for the current array,
        including zebra-stripe, checkerboard, and separated-species patterns. The two
        species-specific targets are written to ``self.target_Rb`` and
        ``self.target_Cs`` and then stacked into the full ``self.target`` tensor.

        Parameters
        ----------
        pattern : Configurations, optional
            Target pattern to generate.
        middle_size : list, optional
            Size of the centered target region. If empty, a default centered region
            is generated from the array size and occupation probability.
        occupation_prob : float, optional
            Filling fraction used when selecting a default centered region.

        Returns
        -------
        None
            This method updates ``self.target_Rb``, ``self.target_Cs``, and
            ``self.target`` in place.
        """
        self.target_Rb = np.zeros([self.shape[0], self.shape[1]], dtype=np.uint8)
        self.target_Cs = np.zeros([self.shape[0], self.shape[1]], dtype=np.uint8)

        if middle_size is None or len(middle_size) == 0:
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
                        self.target_Cs[i, j] = np.uint8(1)
                    else:
                        self.target_Rb[i, j] = np.uint8(1)

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
                        self.target_Cs[i, j] = np.uint8(1)
                    else:
                        self.target_Rb[i, j] = np.uint8(1)

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
                        self.target_Rb[i, j] = np.uint8(1)
                    else:
                        self.target_Cs[i, j] = np.uint8(1)

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
                        self.target_Cs[i, j] = np.uint8(1)
                    else:
                        self.target_Rb[i, j] = np.uint8(1)

        self.target = np.stack([self.target_Rb, self.target_Cs], axis=2, dtype=np.uint8)

    def _apply_moves(self, moves: list) -> Tuple[List, List]:
        """
        Dispatch move application to the species-specific implementation.

        This helper applies a parallel move batch using the state representation
        appropriate for the current simulator configuration. It returns failed move
        bookkeeping in the format produced by the underlying single-species or
        dual-species move-application routine.

        Parameters
        ----------
        moves : list
            Parallel batch of ``Move`` objects to apply.

        Returns
        -------
        list
            Indices of moves that did not succeed.
        list
            Failure flags corresponding to the failed moves.
        """
        if self.n_species == 1:
            return self._apply_moves_single_species(moves)
        elif self.n_species == 2:
            return self._apply_moves_dual_species(moves)

    def _apply_moves_single_species(
        self,
        moves: List[Move],
    ) -> Tuple[List, List]:
        """
        Apply a parallel move batch to a single-species array.

        Moves are classified from the pre-move state using parallel-batch semantics:
        a destination is treated as available if it is vacated by another move in
        the same batch. After classification, successful moves, ejections, and loss
        events are written to ``self.matrix`` in place using the failure state
        already stored on each ``Move``.

        Parameters
        ----------
        moves : list[Move]
            Parallel move batch to apply.

        Returns
        -------
        list
            Indices of moves that did not succeed.
        list
            Failure flags corresponding to the failed moves.

        Raises
        ------
        ValueError
            If the matrix dimensionality is inconsistent with single-species move
            application.
        Exception
            If move application encounters an inconsistent source occupancy relative
            to the classified move outcomes.

        Notes
        -----
        - Move classification is computed from the pre-move state (parallel semantics).
        - Execution is applied to `self.matrix` after classification.
        - Colliding tones are handled via `FailureFlag.LOSS` (set upstream by failure policy).
        """
        state_before = np.array(self.matrix, copy=True)

        n = len(moves)
        if n == 0:
            return [], []

        # Normalize fail_flag from fail_event (robust against stale/default local Move states)
        for mv in moves:
            mv._update_fail_flag()

        # Move uses (row, col) ordering
        from_rows = np.asarray([m.from_row for m in moves], dtype=np.intp)
        from_cols = np.asarray([m.from_col for m in moves], dtype=np.intp)
        to_rows = np.asarray([m.to_row for m in moves], dtype=np.intp)
        to_cols = np.asarray([m.to_col for m in moves], dtype=np.intp)

        if len(self.shape) >= 2:
            n_rows, n_cols = int(self.shape[0]), int(self.shape[1])
        else:
            n_rows, n_cols = int(state_before.shape[0]), int(state_before.shape[1])

        dest_in_bounds = (
            (to_rows >= 0) & (to_rows < n_rows) & (to_cols >= 0) & (to_cols < n_cols)
        )

        # ---- Pre-move occupancy (row, col) ----
        # Supports:
        #   - 2D single-species matrix: (rows, cols)
        #   - 3D single-species matrix: (rows, cols, 1)
        if state_before.ndim == 2:
            occ_before = state_before
        elif state_before.ndim == 3:
            occ_before = np.sum(state_before, axis=-1)  # robust if shape (...,1)
        else:
            raise ValueError(
                f"Unexpected matrix ndim for single-species move application: {state_before.ndim}"
            )

        src_has_atom = occ_before[from_rows, from_cols] > 0

        # Destination occupancy (only for in-bounds destinations)
        dst_occ = np.zeros(n, dtype=np.int64)
        inb_idx = np.where(dest_in_bounds)[0]
        if inb_idx.size > 0:
            dst_occ[inb_idx] = occ_before[to_rows[inb_idx], to_cols[inb_idx]]

        # Parallel semantics: destination is "vacated" if it is a source of any move in this batch
        # (regardless of move outcome, movetype classification is geometry/intended parallel move-set based)
        source_sites = {
            (int(r), int(c)) for r, c in zip(from_rows, from_cols, strict=True)
        }
        dst_vacated_by_batch = np.zeros(n, dtype=bool)
        if inb_idx.size > 0:
            dst_vacated_by_batch[inb_idx] = [
                (int(to_rows[i]), int(to_cols[i])) in source_sites for i in inb_idx
            ]

        # Effective destination occupancy for movetype classification:
        # occupied and NOT vacated by another move in the same batch => illegal
        dst_blocked = (dst_occ > 0) & (~dst_vacated_by_batch)

        # Classify movetypes from pre-move state + parallel batch semantics
        no_atom_mask = ~src_has_atom
        eject_mask = src_has_atom & (~dest_in_bounds)
        legal_mask = src_has_atom & dest_in_bounds & (~dst_blocked)
        illegal_mask = src_has_atom & dest_in_bounds & dst_blocked

        for i in np.where(no_atom_mask)[0]:
            moves[i].movetype = MoveType.NO_ATOM_TO_MOVE
            if int(moves[i].fail_event) == int(FailureEvent.SUCCESS):
                moves[i].set_failure_event(FailureEvent.NO_ATOM)
                moves[i]._update_fail_flag()

        for i in np.where(eject_mask)[0]:
            moves[i].movetype = MoveType.EJECT_MOVE

        for i in np.where(legal_mask)[0]:
            moves[i].movetype = MoveType.LEGAL_MOVE

        for i in np.where(illegal_mask)[0]:
            moves[i].movetype = MoveType.ILLEGAL_MOVE

        # for mv in moves: # moving this 10 lines above
        #     mv._update_fail_flag()

        # is_success = np.asarray([mv.is_successful() for mv in moves], dtype=bool)
        # is_loss = np.asarray([mv.atom_was_lost() for mv in moves], dtype=bool)
        flags_arr = np.asarray([int(mv.fail_flag) for mv in moves], dtype=np.int32)
        movetypes = np.asarray([int(mv.movetype) for mv in moves], dtype=np.int32)
        is_success = flags_arr == int(FailureFlag.SUCCESS)
        is_loss = flags_arr == int(FailureFlag.LOSS)

        failed_moves: list[int] = []
        flags: list[int] = []
        for i in range(n):
            if not is_success[i]:
                failed_moves.append(i)
                flags.append(int(flags_arr[i]))

        # ---- Apply effects to self.matrix (vectorized) ----
        if self.matrix.ndim == 2:
            occ = self.matrix
        else:
            # single-species storage assumed (..., 1)
            occ = self.matrix[:, :, 0]

        is_legal_or_illegal = (movetypes == int(MoveType.LEGAL_MOVE)) | (
            movetypes == int(MoveType.ILLEGAL_MOVE)
        )
        is_eject = movetypes == int(MoveType.EJECT_MOVE)

        # Successful legal/illegal: source -> dest
        idx = np.where(is_success & is_legal_or_illegal)[0]
        if idx.size > 0:
            bad = occ[from_rows[idx], from_cols[idx]] <= 0
            if bad.any():
                j = idx[np.where(bad)[0][0]]
                raise Exception(
                    f"Error in move application: NO atom at source "
                    f"({moves[j].from_row}, {moves[j].from_col}) for successful move."
                )
            np.add.at(occ, (from_rows[idx], from_cols[idx]), -1)
            np.add.at(occ, (to_rows[idx], to_cols[idx]), +1)

        # Successful ejection: remove source only
        idx = np.where(is_success & is_eject)[0]
        if idx.size > 0:
            bad = occ[from_rows[idx], from_cols[idx]] <= 0
            if bad.any():
                j = idx[np.where(bad)[0][0]]
                raise Exception(
                    f"Error occured in MoveType assignment. There is NO atom at "
                    f"({moves[j].from_row}, {moves[j].from_col})."
                )
            np.add.at(occ, (from_rows[idx], from_cols[idx]), -1)

        # Loss failures (including collision_*): remove source only
        idx = np.where(is_loss)[0]
        if idx.size > 0:
            bad = occ[from_rows[idx], from_cols[idx]] <= 0
            if bad.any():
                j = idx[np.where(bad)[0][0]]
                raise Exception(
                    f"Error occured in MoveType. There is NO atom at "
                    f"({moves[j].from_row}, {moves[j].from_col})."
                )
            np.add.at(occ, (from_rows[idx], from_cols[idx]), -1)

        return failed_moves, flags

    def _apply_moves_dual_species(
        self,
        moves: List[Move],
    ) -> Tuple[List, List]:
        """
        Apply a parallel move batch to a dual-species array.

        Moves are classified from the pre-move state using site occupancy and
        parallel-batch semantics. Species identity is inferred from the source site
        before mutation, and successful moves, ejections, and loss events are then
        applied to the appropriate species channel of ``self.matrix``.

        Parameters
        ----------
        moves : list[Move]
            Parallel move batch to apply.

        Returns
        -------
        list
            Indices of moves that did not succeed.
        list
            Failure flags corresponding to the failed moves.
        """
        state_before = np.array(self.matrix, copy=True)

        n = len(moves)
        if n == 0:
            return [], []

        for mv in moves:
            mv._update_fail_flag()

        # Move uses (row, col)
        from_rows = np.asarray([m.from_row for m in moves], dtype=np.intp)
        from_cols = np.asarray([m.from_col for m in moves], dtype=np.intp)
        to_rows = np.asarray([m.to_row for m in moves], dtype=np.intp)
        to_cols = np.asarray([m.to_col for m in moves], dtype=np.intp)

        if hasattr(self, "shape") and len(self.shape) >= 2:
            n_rows, n_cols = int(self.shape[0]), int(self.shape[1])
        else:
            n_rows, n_cols = int(state_before.shape[0]), int(state_before.shape[1])

        dest_in_bounds = (
            (to_rows >= 0) & (to_rows < n_rows) & (to_cols >= 0) & (to_cols < n_cols)
        )

        # Pre-move site occupancy counts (0,1,2 possible)
        src_counts = np.sum(state_before[from_rows, from_cols, :], axis=1)

        dst_counts = np.zeros(n, dtype=np.int64)
        inb_idx = np.where(dest_in_bounds)[0]
        if inb_idx.size > 0:
            dst_counts[inb_idx] = np.sum(
                state_before[to_rows[inb_idx], to_cols[inb_idx], :], axis=1
            )

        # Parallel semantics: destination is "vacated" if it is a source of any move in this batch
        source_sites = {
            (int(r), int(c)) for r, c in zip(from_rows, from_cols, strict=True)
        }
        dst_vacated_by_batch = np.zeros(n, dtype=bool)
        dst_vacated_by_batch[inb_idx] = np.fromiter(
            ((int(to_rows[i]), int(to_cols[i])) in source_sites for i in inb_idx),
            dtype=bool,
            count=inb_idx.size,
        )

        # Destination is blocked only if occupied and not vacated by a batch move
        dst_blocked = (dst_counts > 0) & (~dst_vacated_by_batch)

        no_atom_mask = src_counts == 0
        valid_source_mask = src_counts == 1
        weird_source_mask = src_counts != 1  # conservative handling if malformed state

        eject_mask = valid_source_mask & (~dest_in_bounds)
        legal_mask = valid_source_mask & dest_in_bounds & (~dst_blocked)
        illegal_mask = valid_source_mask & dest_in_bounds & dst_blocked

        for i in np.where(no_atom_mask | weird_source_mask)[0]:
            moves[i].movetype = MoveType.NO_ATOM_TO_MOVE
            if int(moves[i].fail_event) == int(FailureEvent.SUCCESS):
                moves[i].set_failure_event(FailureEvent.NO_ATOM)

        for i in np.where(eject_mask)[0]:
            moves[i].movetype = MoveType.EJECT_MOVE

        for i in np.where(legal_mask)[0]:
            moves[i].movetype = MoveType.LEGAL_MOVE

        for i in np.where(illegal_mask)[0]:
            moves[i].movetype = MoveType.ILLEGAL_MOVE

        for mv in moves:
            mv._update_fail_flag()

        is_success = np.asarray([mv.is_successful() for mv in moves], dtype=bool)
        is_loss = np.asarray([mv.atom_was_lost() for mv in moves], dtype=bool)

        flags_arr = np.asarray([int(mv.fail_flag) for mv in moves], dtype=np.int32)
        movetypes = np.asarray([int(mv.movetype) for mv in moves], dtype=np.int32)

        failed_moves: list[int] = []
        flags: list[int] = []
        for i in range(n):
            if not is_success[i]:
                failed_moves.append(i)
                flags.append(int(flags_arr[i]))

        is_legal_or_illegal = (movetypes == int(MoveType.LEGAL_MOVE)) | (
            movetypes == int(MoveType.ILLEGAL_MOVE)
        )
        is_eject = movetypes == int(MoveType.EJECT_MOVE)

        # Species identity from pre-move state
        src_species0 = state_before[from_rows, from_cols, 0] > 0
        src_species1 = state_before[from_rows, from_cols, 1] > 0

        # Successful legal/illegal
        idx = np.where(is_success & is_legal_or_illegal)[0]
        if idx.size > 0:
            i0 = idx[src_species0[idx]]
            i1 = idx[src_species1[idx]]

            if i0.size > 0:
                np.add.at(self.matrix[:, :, 0], (from_rows[i0], from_cols[i0]), -1)
                np.add.at(self.matrix[:, :, 0], (to_rows[i0], to_cols[i0]), +1)

            if i1.size > 0:
                np.add.at(self.matrix[:, :, 1], (from_rows[i1], from_cols[i1]), -1)
                np.add.at(self.matrix[:, :, 1], (to_rows[i1], to_cols[i1]), +1)

        # Successful ejection
        idx = np.where(is_success & is_eject)[0]
        if idx.size > 0:
            i0 = idx[src_species0[idx]]
            i1 = idx[src_species1[idx]]

            if i0.size > 0:
                np.add.at(self.matrix[:, :, 0], (from_rows[i0], from_cols[i0]), -1)
            if i1.size > 0:
                np.add.at(self.matrix[:, :, 1], (from_rows[i1], from_cols[i1]), -1)

        # Loss failures
        idx = np.where(is_loss)[0]
        if idx.size > 0:
            i0 = idx[src_species0[idx]]
            i1 = idx[src_species1[idx]]

            if i0.size > 0:
                np.add.at(self.matrix[:, :, 0], (from_rows[i0], from_cols[i0]), -1)
            if i1.size > 0:
                np.add.at(self.matrix[:, :, 1], (from_rows[i1], from_cols[i1]), -1)

        return failed_moves, flags
