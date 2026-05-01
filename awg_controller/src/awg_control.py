"""
AWG (Arbitrary Waveform Generator) control utilities for the atommovr pipeline.

Converts logical atom ``Move`` objects into RF ramp commands (``AWGBatch``)
that are executed by Spectrum Instrumentation cards via the spcm DDS interface.

Amplitude unit : percent of full-scale  (sum ≤ 40 % per channel, per manufacturer)
Frequency unit : Hz
Phase unit     : degrees
Time unit      : seconds
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from atommovr.utils.Move import Move
from atommovr.utils.core import PhysicalParams


# ---------------------------------------------------------------------------
# Hardware constants  (source of truth: cli.py + spcm documentation)
# ---------------------------------------------------------------------------

#: Combined amplitude of all simultaneous DDS tones on one output channel
#: must not exceed 40 % of full-scale (manufacturer recommendation).
MAX_AMPLITUDE_PCT_PER_CHANNEL: float = 40.0

#: Total DDS cores on the Spectrum AWG card (21 cores: indices 0-20).
TOTAL_DDS_CORES: int = 21

#: All cores assignable to channel 0 (V / row AOD).  By default cores 0-19.
ALL_CHANNEL_0_CORES: List[int] = list(range(20))

#: Cores exclusive to channel 0 (never flex-assigned to ch1).
#: Cores 8-11 are flex-assignable to ch1; these remain on ch0.
CHANNEL_0_EXCLUSIVE_CORES: List[int] = [
    c for c in ALL_CHANNEL_0_CORES if c not in {8, 9, 10, 11}
]  # [0,1,2,3,4,5,6,7,12,13,14,15,16,17,18,19]

#: Full core set for channel 1 (H / col AOD): flex cores 8-11 + fixed core 20.
CHANNEL_1_FULL_CORES: List[int] = [8, 9, 10, 11, 20]

#: Minimal core set for channel 1: only the fixed core 20.
CHANNEL_1_SINGLE_CORE: List[int] = [20]

#: Default mapping after ``cores_on_channel(1, 8-11, 20)`` has been called.
#: Channel 0 keeps its exclusive cores; channel 1 gets the full set.
CHANNEL_CORE_MAP: Dict[int, List[int]] = {
    0: CHANNEL_0_EXCLUSIVE_CORES,
    1: CHANNEL_1_FULL_CORES,
}


def compute_core_assignments(
    n_row_tones: int,
    n_col_tones: int,
) -> Dict[int, List[int]]:
    """Determine DDS core indices assigned to each output channel.

    The Spectrum card has 21 cores.  Cores 8-11 are flex-assignable between
    channel 0 and channel 1.  Core 20 is fixed to channel 1.  Remaining
    cores (0-7, 12-19) are fixed to channel 0.

    Mirrors ``configure_cores()`` in cli.py (source of truth).

    Parameters
    ----------
    n_row_tones
        Simultaneous tones on channel 0 (V / row AOD).
    n_col_tones
        Simultaneous tones on channel 1 (H / col AOD).

    Returns
    -------
    dict
        ``{0: [core_indices_for_ch0], 1: [core_indices_for_ch1]}``

    Raises
    ------
    ValueError
        If the requested tone counts exceed single-card hardware limits.
    """
    if n_col_tones > len(CHANNEL_1_FULL_CORES):
        raise ValueError(
            f"Channel 1 supports at most {len(CHANNEL_1_FULL_CORES)} tones; "
            f"{n_col_tones} requested."
        )

    # cli.py: if ch1 needs ≤1 tone → only core 20; otherwise full set 8-11, 20
    if n_col_tones <= 1:
        ch1_cores = CHANNEL_1_SINGLE_CORE
        ch0_pool = ALL_CHANNEL_0_CORES           # 0-19 (no overlap)
    else:
        ch1_cores = CHANNEL_1_FULL_CORES
        ch0_pool = CHANNEL_0_EXCLUSIVE_CORES     # 0-7, 12-19

    if n_row_tones > len(ch0_pool):
        raise ValueError(
            f"Channel 0 has {len(ch0_pool)} available cores "
            f"(after ch1 assignment); {n_row_tones} tones requested."
        )

    return {
        0: ch0_pool[:n_row_tones],
        1: ch1_cores[:n_col_tones],
    }


def validate_hardware_limits(grid_rows: int, grid_cols: int) -> None:
    """Raise ``ValueError`` if grid dimensions exceed single-card DDS limits.

    Call this during hardware initialisation to fail fast with a clear message.
    """
    compute_core_assignments(grid_rows, grid_cols)


# ---------------------------------------------------------------------------
# Data-transfer objects
# ---------------------------------------------------------------------------

@dataclass
class AODSettings:
    """Frequency-range and geometry settings for the AWG card.

    ``f_min_v / f_max_v`` span the vertical (row) AOD bandwidth.
    ``f_min_h / f_max_h`` span the horizontal (column) AOD bandwidth.
    Grid dimensions determine the site-to-frequency mapping.
    """

    # Frequency ranges (Hz)
    f_min_v: float = 60e6
    f_max_v: float = 100e6
    f_min_h: float = 60e6
    f_max_h: float = 100e6

    # Total trap-grid dimensions
    grid_rows: int = 10
    grid_cols: int = 10

    # Target sub-array dimensions (informational, used by the controller)
    target_rows: int = 6
    target_cols: int = 6

    alignment: str = "center"   # "center" | "start"

    @property
    def f_spacing_v(self) -> float:
        """Row-axis inter-site frequency step (Hz)."""
        return (self.f_max_v - self.f_min_v) / max(self.grid_rows - 1, 1)

    @property
    def f_spacing_h(self) -> float:
        """Column-axis inter-site frequency step (Hz)."""
        return (self.f_max_h - self.f_min_h) / max(self.grid_cols - 1, 1)

    def validate_core_limits(self) -> None:
        """Raise ``ValueError`` if grid dimensions exceed available DDS cores."""
        compute_core_assignments(self.grid_rows, self.grid_cols)


@dataclass
class RFRamp:
    """Single-tone RF command for one DDS core.

    Attributes
    ----------
    channel : int
        Hardware output channel (0 = V/row AOD, 1 = H/col AOD).
    core : int
        DDS core index assigned to this tone.
    f_start : float
        Pre-move frequency (Hz) — the current trap position.
    f_end : float
        Post-move frequency (Hz) — the target trap position.
        This is the value written to the card for the next trigger.
    amplitude_pct : float
        Per-core amplitude (%).  All ramps on the same channel must sum
        to ≤ MAX_AMPLITUDE_PCT_PER_CHANNEL (40 %).
    phase_deg : float
        Core phase offset (degrees), default 0.
    duration_s : float
        Physical move duration (s).  The controller waits this long before
        acquiring the verification image.
    """

    channel: int
    core: int
    f_start: float
    f_end: float
    amplitude_pct: float
    phase_deg: float = 0.0
    duration_s: float = 0.0


@dataclass
class AWGBatch:
    """Set of RF commands executed in a single hardware trigger event.

    One ``AWGBatch`` corresponds to one parallel batch of atom moves.
    """

    ramps: List[RFRamp]
    total_duration_s: float


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

class RFConverter:
    """Translate logical ``Move`` objects into ``AWGBatch`` hardware commands.

    Amplitude budget rule (from cli.py comments):
        Per-core amplitude = 40 % / n_simultaneous_tones_on_channel

    Move duration is computed from the Chebyshev distance between source and
    target (parallel V and H moves overlap in time) and the AOD speed.

    Parameters
    ----------
    settings : AODSettings
    physical_params : PhysicalParams
    """

    def __init__(self, settings: AODSettings, physical_params: PhysicalParams) -> None:
        self.settings = settings
        self.params = physical_params

        # Compute core assignments.  Falls back to sequential virtual indices
        # when the grid exceeds single-card limits (simulation / testing).
        try:
            self._core_map = compute_core_assignments(
                settings.grid_rows, settings.grid_cols,
            )
        except ValueError:
            self._core_map = {
                0: list(range(settings.grid_rows)),
                1: list(range(settings.grid_cols)),
            }

    @property
    def core_map(self) -> Dict[int, List[int]]:
        """Core-index assignments computed at construction time.

        Used by the controller to call ``dds.cores_on_channel()``.
        """
        return self._core_map

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_freq(self, row: int) -> float:
        return self.settings.f_min_v + row * self.settings.f_spacing_v

    def _col_to_freq(self, col: int) -> float:
        return self.settings.f_min_h + col * self.settings.f_spacing_h

    def _move_duration_s(self, moves: List[Move]) -> float:
        """Physical duration (s) for the longest move in the batch."""
        if not moves:
            return 0.0
        max_cheb = max(
            max(abs(m.to_row - m.from_row), abs(m.to_col - m.from_col))
            for m in moves
        )
        dist_m = max_cheb * self.params.spacing
        duration_s = dist_m / max(self.params.AOD_speed, 1e-15)
        return max(duration_s, 1e-6)   # floor at 1 us

    @staticmethod
    def _per_tone_amplitude(n: int) -> float:
        if n <= 0:
            return MAX_AMPLITUDE_PCT_PER_CHANNEL
        return MAX_AMPLITUDE_PCT_PER_CHANNEL / n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def holding_config(self) -> AWGBatch:
        """Static holding batch: every grid site at its resting frequency.

        All ``grid_rows`` V-cores and ``grid_cols`` H-cores are set with
        ``f_start == f_end`` (no motion).  Sent to the card between
        rearrangement rounds so atoms remain trapped.
        """
        v_cores = self._core_map[0]
        h_cores = self._core_map[1]
        amp_v = self._per_tone_amplitude(len(v_cores))
        amp_h = self._per_tone_amplitude(len(h_cores))

        ramps: List[RFRamp] = []
        for i, core in enumerate(v_cores):
            f = self._row_to_freq(i)
            ramps.append(RFRamp(
                channel=0, core=core,
                f_start=f, f_end=f,
                amplitude_pct=amp_v,
            ))
        for j, core in enumerate(h_cores):
            f = self._col_to_freq(j)
            ramps.append(RFRamp(
                channel=1, core=core,
                f_start=f, f_end=f,
                amplitude_pct=amp_h,
            ))
        return AWGBatch(ramps=ramps, total_duration_s=0.0)

    def convert_moves(self, moves: List[Move]) -> AWGBatch:
        """Convert one parallel move batch into an ``AWGBatch``.

        Every grid row and column is included in the batch so that the
        total amplitude per channel stays constant at 40 %.  Rows and
        columns not involved in any move keep their resting frequency
        (``f_start == f_end``).

        The rearrangement algorithm guarantees that at most one target row
        is assigned per source row (and likewise for columns) within a
        single batch.

        Returns
        -------
        AWGBatch
            Holding batch if *moves* is empty, otherwise a full-grid batch
            with the moving tones updated.

        Raises
        ------
        ValueError
            If two moves assign conflicting targets to the same source
            row or column, or if a target index is out of bounds.
        """
        if not moves:
            return self.holding_config()

        duration_s = self._move_duration_s(moves)
        v_cores = self._core_map[0]
        h_cores = self._core_map[1]
        amp_v = self._per_tone_amplitude(len(v_cores))
        amp_h = self._per_tone_amplitude(len(h_cores))

        # Build source → destination maps for each axis
        row_targets: Dict[int, int] = {}
        col_targets: Dict[int, int] = {}
        for m in moves:
            if m.from_row in row_targets and row_targets[m.from_row] != m.to_row:
                raise ValueError(
                    f"Conflicting row targets: row {m.from_row} → "
                    f"{row_targets[m.from_row]} and {m.to_row}"
                )
            if m.from_col in col_targets and col_targets[m.from_col] != m.to_col:
                raise ValueError(
                    f"Conflicting column targets: col {m.from_col} → "
                    f"{col_targets[m.from_col]} and {m.to_col}"
                )
            row_targets[m.from_row] = m.to_row
            col_targets[m.from_col] = m.to_col

        grid_rows = self.settings.grid_rows
        grid_cols = self.settings.grid_cols

        ramps: List[RFRamp] = []
        for row_idx, core in enumerate(v_cores):
            target_row = row_targets.get(row_idx, row_idx)
            if target_row < 0 or target_row >= grid_rows:
                raise ValueError(
                    f"Row move targets out-of-bounds index {target_row} "
                    f"(grid has {grid_rows} rows)."
                )
            ramps.append(RFRamp(
                channel=0, core=core,
                f_start=self._row_to_freq(row_idx),
                f_end=self._row_to_freq(target_row),
                amplitude_pct=amp_v, duration_s=duration_s,
            ))
        for col_idx, core in enumerate(h_cores):
            target_col = col_targets.get(col_idx, col_idx)
            if target_col < 0 or target_col >= grid_cols:
                raise ValueError(
                    f"Column move targets out-of-bounds index {target_col} "
                    f"(grid has {grid_cols} columns)."
                )
            ramps.append(RFRamp(
                channel=1, core=core,
                f_start=self._col_to_freq(col_idx),
                f_end=self._col_to_freq(target_col),
                amplitude_pct=amp_h, duration_s=duration_s,
            ))

        return AWGBatch(ramps=ramps, total_duration_s=duration_s)

    def convert_sequence(self, move_batches: List[List[Move]]) -> List[AWGBatch]:
        """Convert a full rearrangement sequence into a list of ``AWGBatch`` objects."""
        return [self.convert_moves(b) for b in move_batches]