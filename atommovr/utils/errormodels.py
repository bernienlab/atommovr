# Collection of ErrorModel objects representing various loss processes

from random import random

import numpy as np

from atommovr.utils import Move
from atommovr.utils.core import atom_loss, atom_loss_dual
from atommovr.utils.ErrorModel import ErrorModel


class ZeroNoise(ErrorModel):
    """
    Simulates errorless rearrangement (assumes perfect tweezers
    and an infinitely long vacuum-limited lifetime).
    """

    def __init__(
        self,
        putdown_time: float = 0,
        pickup_time: float = 0,
        accel_time: float = 0,
        decel_time: float = 0,
        seed: int | None = None,
    ):
        super().__init__(
            putdown_time=putdown_time,
            pickup_time=pickup_time,
            accel_time=accel_time,
            decel_time=decel_time,
            pickup_fail_rate=0.0,
            putdown_fail_rate=0.0,
            accel_fail_rate=0.0,
            decel_fail_rate=0.0,
            lifetime=np.inf,
            seed=seed,
        )
        self.name = "ZeroNoise"

    def __repr__(self) -> str:
        return self.name

    def apply_accel_errors_mask(
        self, event_mask: np.ndarray, eligible: np.ndarray
    ) -> None:
        pass

    def apply_decel_errors_mask(
        self, event_mask: np.ndarray, eligible: np.ndarray
    ) -> None:
        pass

    def apply_pickup_errors_mask(
        self, event_mask: np.ndarray, eligible: np.ndarray
    ) -> None:
        pass

    def apply_putdown_errors_mask(
        self, event_mask: np.ndarray, eligible: np.ndarray
    ) -> None:
        pass

    def get_atom_loss(
        self, state: np.ndarray, evolution_time: float, n_species: int = 1
    ) -> tuple[np.ndarray, bool]:
        r"""
        Given the current state of the atom array, applies any general loss process
        over the period $\Delta t$ = evolution_time.

        For this error model, it just returns the same state.

        ## Parameters
        state : np.ndarray
            the current state of the atom array.
        evolution_time : float
            the time over which we calculate the loss process (usually the time
            for a single move or set of parallel moves).
        - n_species : int, optional (default = 1)
            the number of atomic species (single, dual).

        ## Returns
        new_state : np.ndarray
            the state after the loss process.
        loss_flag : bool
            1 if any atom loss occurred, 0 if not.
        """
        if n_species not in [1, 2]:
            raise ValueError(f'Parameter "n_species" must be 1 or 2, not {n_species}')
        loss_flag = False
        new_state = state.copy()
        return new_state, loss_flag

    # def get_move_errors(self, state: np.ndarray, moves: list[Move]) -> list:
    #     """
    #     Given a set of moves and the current state, assigns
    #     an attribute `failure_flag` to the move.
    #     - If the move suceeds, `move_failure_flag = 0`.
    #     - If the move fails but the atom remains in its original position,
    #     `move.failure_flag = 1`.
    #     - If the move fails and the atom is also ejected,
    #     `move_failure_flag = 2`.

    #     For this noise model, it sets all failure flags to 0.
    #     """
    #     for move in moves:
    #         move.failure_flag = 0

    #     return moves


class UniformVacuumTweezerError(ErrorModel):
    """
    Considers atom loss due to imperfect vacuum
    (i.e. collisions with background gas particles)
    and uniform tweezer failure rates.

    ## Parameters
     - `pickup_fail_rate` (optional): float, between 0 and 1.
     Probability that an atom to be moved will be not picked up by the moving tweezer
    (in this case, the atom is not lost but just stays in its original spot). Default
    value is 0.01 (1%).
     - `putdown_fail_rate` (optional): float, between 0 and 1.
    Probability that an atom to be moved will be picked up by the moving tweezer, but
    will be subsequently lost in the transfer to the new tweezer. Default value is
    0.01 (1%).
     - `lifetime` (optional): float.
    Vacuum limited lifetime of an individual atom (assumed to be uniform for all atoms),
    in seconds. Default value is 30.
    """

    def __init__(
        self,
        putdown_time: float = 0,
        pickup_time: float = 0,
        accel_time: float = 0,
        decel_time: float = 0,
        pickup_fail_rate: float = 0.01,
        putdown_fail_rate: float = 0.01,
        accel_fail_rate: float = 0,
        decel_fail_rate: float = 0,
        lifetime: float = 30,
        seed: int | None = None,
    ):
        super().__init__(
            putdown_time=putdown_time,
            pickup_time=pickup_time,
            accel_time=accel_time,
            decel_time=decel_time,
            pickup_fail_rate=pickup_fail_rate,
            putdown_fail_rate=putdown_fail_rate,
            accel_fail_rate=accel_fail_rate,
            decel_fail_rate=decel_fail_rate,
            lifetime=lifetime,
            seed=seed,
        )
        self.name = "UniformVacuumTweezerError"

    def __repr__(self) -> str:
        return self.name

    def get_atom_loss(
        self, state: np.ndarray, evolution_time: float, n_species: int = 1
    ) -> tuple[np.ndarray, bool]:
        r"""
        Given the current state of the atom array, applies any general loss process
        over the period $\Delta t$ = evolution_time.

        For this error model, we consider uniform loss from background gas particles
        knocking atoms out of their traps.

        ## Parameters
        - state (np.ndarray). The current state of the atom array.
        - evolution_time (float). The time over which we calculate the loss process
        (usually the time for a single move).
        - n_species (int, must be 1 or 2). The number of atomic species (single, dual).

        ## Returns
        - new_state (np.ndarray). The state after the loss process.
        - loss_flag (bool). 1 if any atom loss occurred, 0 if not.
        """
        if n_species not in [1, 2]:
            raise ValueError(f'Parameter "n_species" must be 1 or 2, not {n_species}')
        evolution_time = evolution_time
        if n_species == 1:
            new_state, loss_flag = atom_loss(
                state, evolution_time, self.lifetime, self.rng
            )
        elif n_species == 2:
            new_state, loss_flag = atom_loss_dual(
                state, evolution_time, self.lifetime, self.rng
            )
        else:
            raise ValueError(
                f"Parameter 'n_species' must be either 1 or 2, not {n_species}."
            )
        return new_state, loss_flag

        # def get_move_errors(self, state: np.ndarray, moves: list[Move], putdown_events: set, pickup_events: set) -> list[Move]:

    #     """
    #     Given a set of moves and the current state, assigns
    #     an attribute `failure_flag` to the move.
    #     - If the move suceeds, `move_failure_flag = 0`.
    #     - If the move fails but the atom remains in its original position,
    #     `move.failure_flag = 1`.
    #     - If the move fails and the atom is also ejected,
    #     `move_failure_flag = 2`.

    #     In this error model, we uniformly sample from a probability distribution
    #     specified by the class attributes `pickup_fail_rate` and
    #     `putdown_fail_rate`.
    #     """

    #     move_fails = random.choices([0, 1, 2],
    #                                 weights=[1-self.pickup_fail_rate-self.putdown_fail_rate,
    #                                 self.pickup_fail_rate,
    #                                 self.putdown_fail_rate],
    #                                 k = len(moves))
    #     for move_index, move in enumerate(moves):
    #         move.failure_flag = move_fails[move_index]
    #     return moves


class YbRydbergAODErrorModel(ErrorModel):
    """
    Error model for Ytterbium-171 atoms in an AOD-based optical tweezer array,
    incorporating physics relevant to Rydberg platforms.

    References:
    - Barredo et al., Science 354, 1021 (2016)
    - Norcia et al., Phys. Rev. X 13, 041034 (2023)
    - Wilson et al., Phys. Rev. Lett. 128, 033201 (2022)
    - Chen et al., Phys. Rev. A 105, 052438 (2022)
    """

    def __init__(
        self,
        pickup_fail_rate: float = 0.005,
        putdown_fail_rate: float = 0.005,
        accel_time: float = 0,
        decel_time: float = 0,
        accel_fail_rate: float = 0,
        decel_fail_rate: float = 0,
        seed: int | None = None,
        move_distance_penalty: float = 0.001,
        aod_jitter_probability: float = 0.0,
        lifetime: float = 20.0,
        interaction_repulsion: bool = True,
        pickup_time: float = 1e-4,
        putdown_time: float = 1e-4,
    ):
        """
        Parameters:
        - pickup_fail_rate: Base probability of failing to pick up an atom (stays in place).
        - putdown_fail_rate: Base probability of failing to put down an atom (atom lost).
        - move_distance_penalty: Additional failure probability per lattice site unit of distance.
        - aod_jitter_probability: Probability of move failure due to AOD pointing jitter/drifts.
        - lifetime: Vacuum-limited lifetime of 171Yb ground state atoms (seconds).
        - interaction_repulsion: Whether proximate atoms repel/heat (placeholder).
        - pickup_time: Time overhead for pickup (s).
        - putdown_time: Time overhead for putdown (s).
        """
        super().__init__(
            putdown_time=putdown_time,
            pickup_time=pickup_time,
            accel_time=accel_time,
            decel_time=decel_time,
            pickup_fail_rate=pickup_fail_rate,
            putdown_fail_rate=putdown_fail_rate,
            accel_fail_rate=accel_fail_rate,
            decel_fail_rate=decel_fail_rate,
            lifetime=lifetime,
            seed=seed,
        )
        self.name = "YbRydbergAODErrorModel"

        self.move_distance_penalty = move_distance_penalty
        self.aod_jitter_probability = aod_jitter_probability
        self.interaction_repulsion = interaction_repulsion

    def __repr__(self) -> str:
        return f"{self.name}(tau={self.lifetime}s)"

    def get_atom_loss(
        self, state: np.ndarray, evolution_time: float, n_species: int = 1
    ) -> tuple[np.ndarray, bool]:
        r"""
        Given the current state of the atom array, applies any general loss process
        over the period $\Delta t$ = evolution_time.

        For this error model, we consider uniform loss from background gas particles
        knocking atoms out of their traps, a loss from picking up and putting down atoms,
        jitter from AOD pointing errors, and loss from move distances.

        ## Parameters
        - state (np.ndarray). The current state of the atom array.
        - evolution_time (float). The time over which we calculate the loss process
        (usually the time for a single move).
        - n_species (int, must be 1 or 2). The number of atomic species (single, dual).

        ## Returns
        - new_state (np.ndarray). The state after the loss process.
        - loss_flag (bool). 1 if any atom loss occurred, 0 if not.
        """
        if n_species not in [1, 2]:
            raise ValueError(f'Parameter "n_species" must be 1 or 2, not {n_species}')
        evolution_time = evolution_time
        if n_species == 1:
            new_state, loss_flag = atom_loss(
                state, evolution_time, self.lifetime, self.rng
            )
        elif n_species == 2:
            new_state, loss_flag = atom_loss_dual(
                state, evolution_time, self.lifetime, self.rng
            )
        else:
            raise ValueError(
                f"Parameter 'n_species' must be either 1 or 2, not {n_species}."
            )
        return new_state, loss_flag

    def get_move_errors(self, state: np.ndarray, moves: list[Move]) -> list[Move]:
        """
        Calculates failure flags for specific moves based on distance and AOD noise.
        - Failure 0: Success
        - Failure 1: Pickup failed (atom stays)
        - Failure 2: Putdown failed (atom lost)
        """
        for move in moves:
            # Probability of pickup failure (static + jitter)
            p_pickup = self.pickup_fail_rate + self.aod_jitter_probability
            p_pickup = min(max(p_pickup, 0.0), 1.0)

            # Probability of putdown failure (static + jitter + heating from move)
            p_putdown = (
                self.putdown_fail_rate
                + self.aod_jitter_probability
                + (self.move_distance_penalty * move.distance)
            )
            p_putdown = min(max(p_putdown, 0.0), 1.0)

            weights = [1.0 - p_pickup - p_putdown, p_pickup, p_putdown]
            if sum(weights) > 1.0 or weights[0] < 0:
                # Fallback if probs sum > 1
                total = p_pickup + p_putdown
                if total > 0:
                    weights = [0.0, p_pickup / total, p_putdown / total]
                else:
                    weights = [1.0, 0.0, 0.0]

            failure_flag = random.choices([0, 1, 2], weights=weights, k=1)[0]
            move.failure_flag = failure_flag

        return moves
