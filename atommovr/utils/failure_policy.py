"""
Failure event bitmask policy (ordering + suppression rules).

This module defines:
- `FailureEvent`: tracks the physical mechanisms by which a `Move` can fail (see atommovr.utils.Move for info on this class).
- `FailureBit`: bit positions for recording multiple failure events per `Move`.
- `FailureFlag`: tracks the simulation outcome of a move.
- `FAILURE_EVENT_TO_FLAG`: dictionary mapping events to flags for use in simulation.
- `BIT_TO_EVENT`: dictionary mapping bits to primary `FailureEvent`s.
- `PRIMARY_EVENT_ORDER`: precedence order for resolving multiple `FailureEvent`s to one primary `FailureEvent`.
- `bit_value`: returns the integer mask value for a given FailureBit.
- `suppress_inplace`: removes physically irrelevant events from an event bitmask.
- `resolve_primary_events`: converts event bitmasks into primary `FailureEvent` codes.

IMPORTANT:
- Each FailureBit enum value is a *bit position* (not a 1<<value).
- We preserve existing bit positions for backward compatibility with saved masks.
"""

from __future__ import annotations

from enum import IntEnum
import numpy as np

# --- IntEnum classes ---


class FailureEvent(IntEnum):
    """
    Tracks the physical mechanism by which a `Move` can fail.

    Each event corresponds to a specific physical process during atom transport:
    - SUCCESS: Move completed without errors.
    - PICKUP_FAIL: AOD tweezer failed to capture atom from static tweezer.
    - PUTDOWN_FAIL: Atom lost during transfer from AOD to static tweezer.
    - NO_ATOM: No atom present at source position.
    - COLLISION_INEVITABLE: Atom in a static AOD tweezer lost due to collision with a moving AOD tweezer.
    - COLLISION_AVOIDABLE: Atom in a moving AOD tweezer lost due to collision unless not picked up.
    - ACCEL_FAIL: Atom lost during acceleration (heating, etc.).
    - DECEL_FAIL: Atom lost during deceleration (heating, etc.).
    - TRANSPORT_FAIL: Atom lost during transport (not yet implemented).
    """

    SUCCESS = 0
    PICKUP_FAIL = 1
    PUTDOWN_FAIL = 2
    NO_ATOM = 3
    ACCEL_FAIL = 4
    DECEL_FAIL = 5
    TRANSPORT_FAIL = 6  # NB: not yet implemented
    COLLISION_INEVITABLE = 7
    COLLISION_AVOIDABLE = 8


class FailureFlag(IntEnum):
    """
    Tracks the simulation outcome of a `Move` attempt.

    This is derived from FailureEvent and determines what happens to the atom
    and how the simulation state should be updated:
    - SUCCESS: Atom successfully moved to destination
    - NO_PICKUP: Atom remains in source SLM tweezer (pickup failed)
    - LOSS: Atom is lost/ejected from the array
    - NO_ATOM: No atom was present to move
    """

    SUCCESS = 0
    NO_PICKUP = 1
    LOSS = 2
    NO_ATOM = 3


class FailureBit(IntEnum):
    """
    Bit positions for the FailureEvent class (this is an IntEnum and not an IntFlag to enable
    conversion into Numpy arrays for fast operations).

    Notes
    -----
    - SUCCESS is represented by bitmask == 0 (no bits set).
    - Each enum value is a *bit position* (not a 1<<value).
    - Existing bit positions MUST NOT change if you want old saved masks to remain readable.
    """

    # Existing bits (DO NOT renumber)
    PICKUP_FAIL = 0
    PUTDOWN_FAIL = 1
    NO_ATOM = 2
    ACCEL_FAIL = 3
    DECEL_FAIL = 4
    TRANSPORT_FAIL = 5  # NB: not implemented yet
    COLLISION_INEVITABLE = 6  # always lost
    COLLISION_AVOIDABLE = 7  # suppressed by pickup fail


# --- Explicit policy objects ---

# dictionary mapping events (FailureEvent) to flags for use in simulation (FailureFlag)
FAILURE_EVENT_TO_FLAG: dict[FailureEvent, FailureFlag] = {
    FailureEvent.SUCCESS: FailureFlag.SUCCESS,
    FailureEvent.NO_ATOM: FailureFlag.NO_ATOM,
    FailureEvent.PICKUP_FAIL: FailureFlag.NO_PICKUP,
    FailureEvent.PUTDOWN_FAIL: FailureFlag.LOSS,
    FailureEvent.ACCEL_FAIL: FailureFlag.LOSS,
    FailureEvent.DECEL_FAIL: FailureFlag.LOSS,
    FailureEvent.TRANSPORT_FAIL: FailureFlag.LOSS,
    FailureEvent.COLLISION_INEVITABLE: FailureFlag.LOSS,
    FailureEvent.COLLISION_AVOIDABLE: FailureFlag.LOSS,
}
# Map bits to a primary FailureEvent.
BIT_TO_EVENT: dict[FailureBit, FailureEvent] = {
    FailureBit.PICKUP_FAIL: FailureEvent.PICKUP_FAIL,
    FailureBit.PUTDOWN_FAIL: FailureEvent.PUTDOWN_FAIL,
    FailureBit.NO_ATOM: FailureEvent.NO_ATOM,
    FailureBit.COLLISION_INEVITABLE: FailureEvent.COLLISION_INEVITABLE,
    FailureBit.COLLISION_AVOIDABLE: FailureEvent.COLLISION_AVOIDABLE,
    FailureBit.ACCEL_FAIL: FailureEvent.ACCEL_FAIL,
    FailureBit.DECEL_FAIL: FailureEvent.DECEL_FAIL,
    FailureBit.TRANSPORT_FAIL: FailureEvent.TRANSPORT_FAIL,
}

# Highest precedence first.
PRIMARY_EVENT_ORDER: list[FailureBit] = [
    FailureBit.NO_ATOM,
    FailureBit.COLLISION_INEVITABLE,
    FailureBit.PICKUP_FAIL,
    FailureBit.COLLISION_AVOIDABLE,
    FailureBit.ACCEL_FAIL,
    FailureBit.TRANSPORT_FAIL,
    FailureBit.DECEL_FAIL,
    FailureBit.PUTDOWN_FAIL,
]


# --- Functions for getting integer masks ---


def bit_value(bit: FailureBit) -> np.uint64:
    """Return the integer mask value for a given FailureBit."""
    return np.uint64(1) << np.uint64(int(bit))


# --- Precomputed bit masks (avoid repeated allocations / loops at runtime) ---

# Internal cache of bit masks for speed; external callers should use bit_value().
_BITMASK: dict[FailureBit, np.uint64] = {b: bit_value(b) for b in FailureBit}


def _or_mask(bits: tuple[FailureBit, ...]) -> np.uint64:
    m = np.uint64(0)
    for b in bits:
        m |= _BITMASK[b]
    return m


# --- Suppression / dominance rules (explicit and future-proof) ---

# Dominance: keep only DOMINANT_BIT when it is set, unless any exception bits are also set.
# Ordering matters: earlier dominance rules run first.
_DOMINANCE_RULES: tuple[tuple[FailureBit, tuple[FailureBit, ...]], ...] = (
    # NO_ATOM dominates everything (no exceptions)
    (FailureBit.NO_ATOM, ()),
    # COLLISION_INEVITABLE dominates everything except NO_ATOM
    (FailureBit.COLLISION_INEVITABLE, (FailureBit.NO_ATOM,)),
    # COLLISION_AVOIDABLE dominates unless NO_ATOM / CROSSED_STATIC / PICKUP_FAIL also occurred
    (
        FailureBit.COLLISION_AVOIDABLE,
        (FailureBit.NO_ATOM, FailureBit.COLLISION_INEVITABLE, FailureBit.PICKUP_FAIL),
    ),
)

# Suppression: if TRIGGER_BIT is set, clear exactly these bits (and nothing else).
_SUPPRESSION_RULES: tuple[tuple[FailureBit, tuple[FailureBit, ...]], ...] = (
    (
        FailureBit.PICKUP_FAIL,
        (
            FailureBit.COLLISION_AVOIDABLE,
            FailureBit.ACCEL_FAIL,
            FailureBit.TRANSPORT_FAIL,
            FailureBit.DECEL_FAIL,
            FailureBit.PUTDOWN_FAIL,
        ),
    ),
    (  # NOTE: although a collision with another tweezer may happen temporally later than other events,
        #       it is assumed to be completely destructive, whereas transport-related failures may (in
        #       later atommovr version) be nondestructive (but cause effects like heating).
        FailureBit.COLLISION_AVOIDABLE,
        (
            FailureBit.ACCEL_FAIL,
            FailureBit.TRANSPORT_FAIL,
            FailureBit.DECEL_FAIL,
            FailureBit.PUTDOWN_FAIL,
        ),
    ),
    (
        FailureBit.ACCEL_FAIL,
        (
            FailureBit.TRANSPORT_FAIL,
            FailureBit.DECEL_FAIL,
            FailureBit.PUTDOWN_FAIL,
        ),
    ),
    (
        FailureBit.TRANSPORT_FAIL,
        (
            FailureBit.DECEL_FAIL,
            FailureBit.PUTDOWN_FAIL,
        ),
    ),
    (
        FailureBit.DECEL_FAIL,
        (FailureBit.PUTDOWN_FAIL,),
    ),
)

# Precompute masks for fast application
_DOMINANCE_EXCEPT_MASK: dict[FailureBit, np.uint64] = {
    dominant: _or_mask(excepts) for dominant, excepts in _DOMINANCE_RULES
}
_SUPPRESS_CLEAR_MASK: dict[FailureBit, np.uint64] = {
    trigger: _or_mask(clears) for trigger, clears in _SUPPRESSION_RULES
}


def suppress_inplace(event_mask: np.ndarray) -> None:
    """
    Apply suppression/dominance rules to `event_mask` in-place.

    Design goals
    ------------
    - Rules are explicit: adding a new FailureBit will NOT be suppressed and will NOT
      suppress others unless it is explicitly included in _DOMINANCE_RULES or _SUPPRESSION_RULES.
    - Dominance rules express "this mechanism makes other recorded mechanisms irrelevant"
      (optionally with exceptions).
    - Suppression rules express "if X happened, ignore these downstream motion-related bits".

    Performance note
    ----------------
    We first compute a batch-level OR-reduction of all bits present. If a rule’s trigger bit
    does not appear anywhere in the batch, we skip evaluating that rule entirely.
    """
    if event_mask.size == 0:
        return
    if event_mask.ndim != 1:
        raise ValueError(f"Parameter 'event_mask' must be 1D, not {event_mask.ndim}D.")

    # We operate in uint64 for predictable bit ops.
    # If event_mask isn't uint64, we still write results back into it.
    m = event_mask.astype(np.uint64, copy=False)

    # Fast batch-level “which bits exist at all?” check.
    present_bits = np.bitwise_or.reduce(m)

    # 1) Apply dominance rules in order: keep only the dominant bit when it applies.
    for dominant, _excepts in _DOMINANCE_RULES:
        dom_bit = _BITMASK[dominant]
        if (present_bits & dom_bit) == 0:
            continue

        hit = (m & dom_bit) != 0

        except_mask = _DOMINANCE_EXCEPT_MASK[dominant]
        if except_mask != 0:
            hit &= (m & except_mask) == 0

        if hit.any():
            m[hit] = dom_bit

    # Dominance can clear many bits; refresh the batch-level present-bit mask once.
    present_bits = np.bitwise_or.reduce(m)

    # 2) Apply suppression rules (these do NOT "keep only trigger"; they only clear specific bits).
    for trigger, _clears in _SUPPRESSION_RULES:
        trig_bit = _BITMASK[trigger]
        if (present_bits & trig_bit) == 0:
            continue

        hit = (m & trig_bit) != 0
        if hit.any():
            m[hit] &= ~_SUPPRESS_CLEAR_MASK[trigger]

    # 3) If we had to upcast, write back.
    # NOTE: this preserves the caller-visible mutation, but the dtype stays as originally provided.
    if m is not event_mask:
        event_mask[:] = m


def suppress_inplace_slow(event_mask: np.ndarray) -> None:
    """
    Apply suppression/dominance rules to `event_mask` in-place.

    Design goals
    ------------
    - Rules are explicit: adding a new FailureBit will NOT be suppressed and will NOT
      suppress others unless it is explicitly included in _DOMINANCE_RULES or _SUPPRESSION_RULES.
    - Dominance rules express "this mechanism makes other recorded mechanisms irrelevant"
      (optionally with exceptions).
    - Suppression rules express "if X happened, ignore these downstream motion-related bits".
    """
    if event_mask.size == 0:
        return
    if event_mask.ndim != 1:
        raise ValueError(f"Parameter 'event_mask' must be 1D, not {event_mask.ndim}D.")

    m = event_mask.astype(np.uint64, copy=False)

    for dominant, _excepts in _DOMINANCE_RULES:
        dom_bit = _BITMASK[dominant]
        hit = (m & dom_bit) != 0
        if not hit.any():
            continue

        except_mask = _DOMINANCE_EXCEPT_MASK[dominant]
        if except_mask != 0:
            hit &= (m & except_mask) == 0
            if not hit.any():
                continue

        m[hit] = dom_bit

    for trigger, _clears in _SUPPRESSION_RULES:
        trig_bit = _BITMASK[trigger]
        hit = (m & trig_bit) != 0
        if not hit.any():
            continue
        m[hit] &= ~_SUPPRESS_CLEAR_MASK[trigger]

    if m is not event_mask:
        event_mask[:] = m


def resolve_primary_events(event_mask: np.ndarray) -> np.ndarray:
    """
    Resolve one primary FailureEvent per move using PRIMARY_EVENT_ORDER.

    Parameters
    ----------
    event_mask : numpy.ndarray
        Integer array of shape (N,) encoding per-move event bitmasks.

    Returns
    -------
    primary : numpy.ndarray
        Integer array of shape (N,) containing FailureEvent codes.
        If a move has no bits set, it returns FailureEvent.SUCCESS.

    Notes
    -----
    This expects `suppress_inplace` has already been applied; if you call
    resolve without suppression, precedence still works but you may encode
    physically irrelevant combinations.
    """
    n = event_mask.size
    primary = np.full(n, int(FailureEvent.SUCCESS), dtype=np.int32)
    if n == 0:
        return primary
    if event_mask.ndim != 1:
        raise ValueError(f"Parameter 'event_mask' must be 1D, not {event_mask.ndim}D.")

    # Ensure unsigned for predictable bit ops (view)
    m = event_mask.astype(np.uint64, copy=False)

    unresolved = m != 0
    for bit in PRIMARY_EVENT_ORDER:
        if not np.any(unresolved):
            break
        bv = bit_value(bit)
        hit = unresolved & ((m & bv) != 0)
        if np.any(hit):
            primary[hit] = int(BIT_TO_EVENT[bit])
            unresolved[hit] = False

    return primary
