"""
End-to-end pipeline tests for the atommovr controller.
========================================================

Tests the full pipeline from artificial atom images through rearrangement
algorithm selection, RF conversion, and AWG batch generation — everything
except actual hardware I/O (spcm is not required).

Follows the pattern established in ``test_algorithms.py``:
- Build deterministic atom arrays with known occupancy
- Run algorithm → moves → RF batches
- Verify move correctness, frequency mapping, amplitude budgets,
  core assignments, and batch structure.
"""

import math
import os
import sys

import numpy as np
import pytest

# Ensure repo root is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Path to controller scripts
_SCRIPTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "scripts"
)

from atommovr.algorithms.single_species import (
    BCv2,
    BalanceAndCompact,
    GeneralizedBalance,
    Hungarian,
    PCFA,
    ParallelHungarian,
    ParallelLBAP,
    Tetris,
)
from atommovr.utils.AtomArray import AtomArray
from atommovr.utils.Move import Move
from awg_controller.src.awg_control import (
    ALL_CHANNEL_0_CORES,
    AWGBatch,
    CHANNEL_0_EXCLUSIVE_CORES,
    CHANNEL_1_FULL_CORES,
    CHANNEL_1_SINGLE_CORE,
    CHANNEL_CORE_MAP,
    MAX_AMPLITUDE_PCT_PER_CHANNEL,
    AODSettings,
    RFConverter,
    RFRamp,
    compute_core_assignments,
    validate_hardware_limits,
)
from atommovr.utils.core import Configurations, PhysicalParams
from awg_controller.src.dds_strategies import (
    MAX_SAFE_TRIGGER_LEVEL_V,
    CameraTriggerConfig,
    DDSCameraTriggeredStrategy,
    DDSPatternStrategy,
    DDSRampStrategy,
    DDSStrategy,
    DDSStreamingStrategy,
    PatternConfig,
    RampConfig,
    STRATEGY_REGISTRY,
    get_strategy,
    _group_ramps_by_channel,
)

# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_algorithms.py style)
# ---------------------------------------------------------------------------

def _centered_target_mask(
    array_shape: tuple[int, int],
    target_rows: int,
    target_cols: int | None = None,
) -> np.ndarray:
    """Create a centred rectangular target mask."""
    if target_cols is None:
        target_cols = target_rows
    mask = np.zeros(array_shape, dtype=int)
    r0 = max(0, (array_shape[0] - target_rows) // 2)
    c0 = max(0, (array_shape[1] - target_cols) // 2)
    mask[r0 : r0 + target_rows, c0 : c0 + target_cols] = 1
    return mask


def _default_source_state(
    array_shape: tuple[int, int],
    target_size: int,
) -> np.ndarray:
    """L-shaped border fill with enough atoms to cover the target."""
    rows, cols = array_shape
    t = target_size
    state = np.zeros(array_shape, dtype=int)
    band_rows = min(rows, t + 2)
    band_cols = min(cols, t + 2)
    state[:band_rows, :] = 1
    state[:, :band_cols] = 1
    state[:, -band_cols:] = 1
    return state


def _build_array_with_target(
    grid_rows: int,
    grid_cols: int,
    target_rows: int,
    target_cols: int,
) -> AtomArray:
    """Build an AtomArray with atoms surrounding a centred target region."""
    shape = (grid_rows, grid_cols)
    target = _centered_target_mask(shape, target_rows, target_cols)
    state = _default_source_state(shape, max(target_rows, target_cols))
    # Clear atoms inside the target so the algorithm must move them in
    state[target == 1] = 0

    arr = AtomArray(list(shape), n_species=1)
    arr.matrix[:, :, 0] = state
    arr.target = target.reshape(grid_rows, grid_cols, 1)
    return arr


def _make_simple_settings(
    grid_rows: int = 10,
    grid_cols: int = 5,
    target_rows: int = 6,
    target_cols: int = 5,
) -> AODSettings:
    return AODSettings(
        f_min_v=60e6,
        f_max_v=100e6,
        f_min_h=60e6,
        f_max_h=100e6,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        target_rows=target_rows,
        target_cols=target_cols,
    )


# =====================================================================
# 1. Core assignment tests
# =====================================================================

class TestCoreAssignments:
    """Verify that compute_core_assignments mirrors cli.py configure_cores."""

    def test_single_col_tone_uses_core_20_only(self):
        """≤1 column tone → only core 20 on ch1, full ch0 pool (0-19)."""
        mapping = compute_core_assignments(n_row_tones=10, n_col_tones=1)
        assert mapping[1] == [20]
        assert mapping[0] == list(range(10))  # first 10 of 0-19

    def test_multi_col_tones_use_flex_cores(self):
        """2-5 column tones → ch1 gets cores 8-11 + 20."""
        mapping = compute_core_assignments(n_row_tones=10, n_col_tones=5)
        assert mapping[1] == [8, 9, 10, 11, 20]
        # ch0 should be the exclusive set (0-7, 12-19), first 10
        assert mapping[0] == CHANNEL_0_EXCLUSIVE_CORES[:10]

    def test_max_ch0_cores_with_single_ch1(self):
        """Up to 20 ch0 tones when ch1 has only 1 tone (core 20)."""
        mapping = compute_core_assignments(n_row_tones=20, n_col_tones=1)
        assert len(mapping[0]) == 20
        assert mapping[1] == [20]

    def test_max_ch0_cores_with_full_ch1(self):
        """Up to 16 exclusive ch0 cores when ch1 uses flex block."""
        mapping = compute_core_assignments(n_row_tones=16, n_col_tones=5)
        assert len(mapping[0]) == 16
        assert mapping[0] == CHANNEL_0_EXCLUSIVE_CORES

    def test_too_many_ch1_tones_raises(self):
        with pytest.raises(ValueError, match="Channel 1"):
            compute_core_assignments(n_row_tones=5, n_col_tones=6)

    def test_too_many_ch0_tones_raises(self):
        with pytest.raises(ValueError, match="Channel 0"):
            compute_core_assignments(n_row_tones=17, n_col_tones=5)

    def test_validate_hardware_limits_accepts_valid(self):
        validate_hardware_limits(10, 5)  # should not raise

    def test_validate_hardware_limits_rejects_invalid(self):
        with pytest.raises(ValueError):
            validate_hardware_limits(20, 6)

    def test_no_core_overlap_between_channels(self):
        """Cores assigned to ch0 and ch1 must be disjoint."""
        for n_col in [1, 3, 5]:
            mapping = compute_core_assignments(n_row_tones=10, n_col_tones=n_col)
            overlap = set(mapping[0]) & set(mapping[1])
            assert overlap == set(), (
                f"Core overlap {overlap} at n_col={n_col}"
            )


# =====================================================================
# 2. RFConverter unit tests
# =====================================================================

class TestRFConverter:
    """Verify frequency mapping, amplitude budgets, and batch structure."""

    @pytest.fixture
    def converter_10x5(self) -> RFConverter:
        settings = _make_simple_settings(grid_rows=10, grid_cols=5)
        return RFConverter(settings, PhysicalParams())

    @pytest.fixture
    def converter_4x3(self) -> RFConverter:
        settings = _make_simple_settings(grid_rows=4, grid_cols=3, target_rows=2, target_cols=2)
        return RFConverter(settings, PhysicalParams())

    # -- frequency mapping --

    def test_row_to_freq_boundaries(self, converter_10x5: RFConverter):
        assert converter_10x5._row_to_freq(0) == pytest.approx(60e6)
        assert converter_10x5._row_to_freq(9) == pytest.approx(100e6)

    def test_col_to_freq_boundaries(self, converter_10x5: RFConverter):
        assert converter_10x5._col_to_freq(0) == pytest.approx(60e6)
        assert converter_10x5._col_to_freq(4) == pytest.approx(100e6)

    def test_freq_spacing(self, converter_10x5: RFConverter):
        s = converter_10x5.settings
        expected_v = (100e6 - 60e6) / 9
        expected_h = (100e6 - 60e6) / 4
        assert s.f_spacing_v == pytest.approx(expected_v)
        assert s.f_spacing_h == pytest.approx(expected_h)

    # -- holding config --

    def test_holding_config_ramp_count(self, converter_10x5: RFConverter):
        """Holding config must emit one ramp per grid site."""
        batch = converter_10x5.holding_config()
        assert isinstance(batch, AWGBatch)
        assert len(batch.ramps) == 10 + 5  # grid_rows + grid_cols
        assert batch.total_duration_s == 0.0

    def test_holding_config_all_static(self, converter_10x5: RFConverter):
        """All ramps in holding config must have f_start == f_end."""
        batch = converter_10x5.holding_config()
        for ramp in batch.ramps:
            assert ramp.f_start == ramp.f_end, (
                f"Core {ramp.core} on ch{ramp.channel}: "
                f"f_start={ramp.f_start} != f_end={ramp.f_end}"
            )

    def test_holding_amplitude_budget(self, converter_10x5: RFConverter):
        """Total amplitude per channel must not exceed 40%."""
        batch = converter_10x5.holding_config()
        ch0_total = sum(r.amplitude_pct for r in batch.ramps if r.channel == 0)
        ch1_total = sum(r.amplitude_pct for r in batch.ramps if r.channel == 1)
        assert ch0_total == pytest.approx(MAX_AMPLITUDE_PCT_PER_CHANNEL)
        assert ch1_total == pytest.approx(MAX_AMPLITUDE_PCT_PER_CHANNEL)

    def test_holding_uses_correct_cores(self, converter_10x5: RFConverter):
        """Each ramp must use a core from the correct channel."""
        batch = converter_10x5.holding_config()
        core_map = converter_10x5.core_map
        for ramp in batch.ramps:
            assert ramp.core in core_map[ramp.channel], (
                f"Core {ramp.core} not in ch{ramp.channel} map {core_map[ramp.channel]}"
            )

    # -- convert_moves --

    def test_empty_moves_returns_holding(self, converter_10x5: RFConverter):
        batch = converter_10x5.convert_moves([])
        holding = converter_10x5.holding_config()
        assert len(batch.ramps) == len(holding.ramps)
        assert batch.total_duration_s == 0.0

    def test_single_move_batch_structure(self, converter_4x3: RFConverter):
        """A single move should still include ALL grid tones."""
        move = Move(from_row=0, from_col=0, to_row=1, to_col=1)
        batch = converter_4x3.convert_moves([move])
        # 4 rows + 3 cols = 7 ramps total
        assert len(batch.ramps) == 4 + 3
        assert batch.total_duration_s > 0

    def test_moving_tone_frequency_changes(self, converter_4x3: RFConverter):
        """Moving tone must have f_end != f_start (unless same row/col)."""
        move = Move(from_row=0, from_col=0, to_row=2, to_col=1)
        batch = converter_4x3.convert_moves([move])
        # Find the row-0 ramp (ch0, should move to row 2)
        row_ramps = [r for r in batch.ramps if r.channel == 0]
        # The core corresponding to row 0 should have f_end = freq(row 2)
        row0_ramp = row_ramps[0]  # first core = row 0
        assert row0_ramp.f_start == pytest.approx(converter_4x3._row_to_freq(0))
        assert row0_ramp.f_end == pytest.approx(converter_4x3._row_to_freq(2))

    def test_non_moving_tones_stay_static(self, converter_4x3: RFConverter):
        """Non-moving row/col tones must have f_start == f_end."""
        move = Move(from_row=0, from_col=0, to_row=2, to_col=1)
        batch = converter_4x3.convert_moves([move])
        for ramp in batch.ramps:
            if ramp.channel == 0 and ramp.core != converter_4x3.core_map[0][0]:
                assert ramp.f_start == ramp.f_end, (
                    f"Non-moving ch0 core {ramp.core} changed frequency"
                )

    def test_amplitude_budget_during_moves(self, converter_4x3: RFConverter):
        """Amplitude sum per channel must be exactly 40% during moves."""
        move = Move(from_row=1, from_col=2, to_row=3, to_col=0)
        batch = converter_4x3.convert_moves([move])
        ch0_total = sum(r.amplitude_pct for r in batch.ramps if r.channel == 0)
        ch1_total = sum(r.amplitude_pct for r in batch.ramps if r.channel == 1)
        assert ch0_total == pytest.approx(MAX_AMPLITUDE_PCT_PER_CHANNEL)
        assert ch1_total == pytest.approx(MAX_AMPLITUDE_PCT_PER_CHANNEL)

    def test_conflicting_row_targets_raise(self, converter_4x3: RFConverter):
        """Two moves sending the same source row to different target rows."""
        moves = [
            Move(from_row=0, from_col=0, to_row=1, to_col=0),
            Move(from_row=0, from_col=1, to_row=2, to_col=1),
        ]
        with pytest.raises(ValueError, match="Conflicting row"):
            converter_4x3.convert_moves(moves)

    def test_conflicting_col_targets_raise(self, converter_4x3: RFConverter):
        """Two moves sending the same source col to different target cols."""
        moves = [
            Move(from_row=0, from_col=0, to_row=0, to_col=1),
            Move(from_row=1, from_col=0, to_row=1, to_col=2),
        ]
        with pytest.raises(ValueError, match="Conflicting column"):
            converter_4x3.convert_moves(moves)

    def test_out_of_bounds_move_raises(self, converter_4x3: RFConverter):
        """Target index beyond grid dimensions should raise."""
        move = Move(from_row=0, from_col=0, to_row=10, to_col=0)
        with pytest.raises(ValueError, match="out-of-bounds"):
            converter_4x3.convert_moves([move])

    def test_convert_sequence(self, converter_4x3: RFConverter):
        batches = converter_4x3.convert_sequence([
            [Move(0, 0, 1, 1)],
            [Move(2, 2, 3, 0)],
            [],
        ])
        assert len(batches) == 3
        assert batches[0].total_duration_s > 0
        assert batches[1].total_duration_s > 0
        assert batches[2].total_duration_s == 0  # empty → holding

    # -- core map simulation fallback --

    def test_core_map_fallback_for_large_grids(self):
        """Grids exceeding hardware limits use virtual sequential indices."""
        large = _make_simple_settings(grid_rows=25, grid_cols=8)
        conv = RFConverter(large, PhysicalParams())
        assert conv.core_map[0] == list(range(25))
        assert conv.core_map[1] == list(range(8))


# =====================================================================
# 3. Move duration tests
# =====================================================================

class TestMoveDuration:
    """Verify that move duration is computed from Chebyshev distance."""

    @pytest.fixture
    def converter(self) -> RFConverter:
        settings = _make_simple_settings(grid_rows=10, grid_cols=10)
        params = PhysicalParams()
        return RFConverter(settings, params)

    def test_diagonal_move_chebyshev(self, converter: RFConverter):
        """A (2,3) move should use Chebyshev dist = max(2,3) = 3."""
        move = Move(0, 0, 2, 3)
        dur = converter._move_duration_s([move])
        expected_dist = 3 * converter.params.spacing
        expected_dur = expected_dist / converter.params.AOD_speed
        assert dur == pytest.approx(max(expected_dur, 1e-6))

    def test_empty_moves_zero_duration(self, converter: RFConverter):
        assert converter._move_duration_s([]) == 0.0

    def test_multiple_moves_takes_longest(self, converter: RFConverter):
        """Duration is determined by the move with largest Chebyshev dist."""
        short = Move(0, 0, 1, 0)  # Chebyshev = 1
        long_ = Move(0, 0, 5, 5)  # Chebyshev = 5
        dur_both = converter._move_duration_s([short, long_])
        dur_long = converter._move_duration_s([long_])
        assert dur_both == pytest.approx(dur_long)


# =====================================================================
# 4. Full pipeline: algorithm → RF conversion
# =====================================================================

PIPELINE_CASES = [
    {"name": "Hungarian",     "cls": Hungarian,     "target_size": 4},
    {"name": "PCFA",          "cls": PCFA,          "target_size": 4},
    {"name": "BCv2",          "cls": BCv2,          "target_size": 4},
    {"name": "Tetris",        "cls": Tetris,        "target_size": 4},
    {"name": "BalanceAndCompact", "cls": BalanceAndCompact, "target_size": 4},
]


@pytest.mark.parametrize("case", PIPELINE_CASES, ids=lambda c: c["name"])
class TestAlgorithmToRFPipeline:
    """Run algorithm → convert moves → validate RF batches end to end."""

    def test_rf_batches_respect_amplitude_budget(self, case):
        """Every generated batch must stay within the 40% budget."""
        target_size = case["target_size"]
        grid_rows, grid_cols = target_size + 2, target_size + 2

        settings = _make_simple_settings(
            grid_rows=grid_rows, grid_cols=grid_cols,
            target_rows=target_size, target_cols=target_size,
        )
        converter = RFConverter(settings, PhysicalParams())
        arr = _build_array_with_target(grid_rows, grid_cols, target_size, target_size)

        algo = case["cls"]()
        _, move_batches, success = algo.get_moves(arr, do_ejection=False)
        assert success, f"{case['name']} reported failure"

        rf_batches = converter.convert_sequence(move_batches)
        for i, batch in enumerate(rf_batches):
            ch0 = sum(r.amplitude_pct for r in batch.ramps if r.channel == 0)
            ch1 = sum(r.amplitude_pct for r in batch.ramps if r.channel == 1)
            assert ch0 == pytest.approx(MAX_AMPLITUDE_PCT_PER_CHANNEL, abs=0.01), (
                f"Batch {i}: ch0 amp {ch0:.2f}% != {MAX_AMPLITUDE_PCT_PER_CHANNEL}%"
            )
            assert ch1 == pytest.approx(MAX_AMPLITUDE_PCT_PER_CHANNEL, abs=0.01), (
                f"Batch {i}: ch1 amp {ch1:.2f}% != {MAX_AMPLITUDE_PCT_PER_CHANNEL}%"
            )

    def test_all_ramps_use_valid_cores(self, case):
        """Every ramp core index must belong to its channel's assignment."""
        target_size = case["target_size"]
        grid_rows, grid_cols = target_size + 2, target_size + 2

        settings = _make_simple_settings(
            grid_rows=grid_rows, grid_cols=grid_cols,
            target_rows=target_size, target_cols=target_size,
        )
        converter = RFConverter(settings, PhysicalParams())
        core_map = converter.core_map
        arr = _build_array_with_target(grid_rows, grid_cols, target_size, target_size)

        algo = case["cls"]()
        _, move_batches, success = algo.get_moves(arr, do_ejection=False)
        assert success

        rf_batches = converter.convert_sequence(move_batches)
        for batch in rf_batches:
            for ramp in batch.ramps:
                assert ramp.core in core_map[ramp.channel], (
                    f"Core {ramp.core} not in ch{ramp.channel} map"
                )

    def test_ramp_count_covers_full_grid(self, case):
        """Each batch must emit exactly grid_rows + grid_cols ramps."""
        target_size = case["target_size"]
        grid_rows, grid_cols = target_size + 2, target_size + 2

        settings = _make_simple_settings(
            grid_rows=grid_rows, grid_cols=grid_cols,
            target_rows=target_size, target_cols=target_size,
        )
        converter = RFConverter(settings, PhysicalParams())
        arr = _build_array_with_target(grid_rows, grid_cols, target_size, target_size)

        algo = case["cls"]()
        _, move_batches, success = algo.get_moves(arr, do_ejection=False)
        assert success

        rf_batches = converter.convert_sequence(move_batches)
        for batch in rf_batches:
            assert len(batch.ramps) == grid_rows + grid_cols

    def test_frequencies_within_aod_bandwidth(self, case):
        """All generated frequencies must stay within [f_min, f_max]."""
        target_size = case["target_size"]
        grid_rows, grid_cols = target_size + 2, target_size + 2

        settings = _make_simple_settings(
            grid_rows=grid_rows, grid_cols=grid_cols,
            target_rows=target_size, target_cols=target_size,
        )
        converter = RFConverter(settings, PhysicalParams())
        arr = _build_array_with_target(grid_rows, grid_cols, target_size, target_size)

        algo = case["cls"]()
        _, move_batches, success = algo.get_moves(arr, do_ejection=False)
        assert success

        rf_batches = converter.convert_sequence(move_batches)
        for batch in rf_batches:
            for ramp in batch.ramps:
                if ramp.channel == 0:
                    assert settings.f_min_v <= ramp.f_end <= settings.f_max_v, (
                        f"V-AOD freq {ramp.f_end/1e6:.2f} MHz OOB"
                    )
                else:
                    assert settings.f_min_h <= ramp.f_end <= settings.f_max_h, (
                        f"H-AOD freq {ramp.f_end/1e6:.2f} MHz OOB"
                    )


# =====================================================================
# 5. Controller simulation tests (no spcm hardware)
# =====================================================================

class TestControllerSimulation:
    """Test the atommovrController in simulation mode (spcm unavailable)."""

    @pytest.fixture
    def controller(self):
        """Import and create a controller in simulation mode."""
        # The controller file guards spcm with try/except and sets
        # _HW_AVAILABLE = False, running in SIM mode.
        sys.path.insert(0, _SCRIPTS_DIR)

        from atommovr_controller import (
            atommovrController,
            HardwareConfig,
            SoftwareConfig,
        )

        sw = SoftwareConfig(
            grid_size=8,
            target_size=4,
            algorithm_name="PCFA",
            max_rounds=3,
            aod_settings=_make_simple_settings(
                grid_rows=8, grid_cols=5, target_rows=4, target_cols=4,
            ),
        )
        hw = HardwareConfig()
        ctrl = atommovrController(sw, hw)
        yield ctrl
        ctrl.shutdown()

    def test_controller_creates_rf_converter(self, controller):
        """Controller must initialise an RFConverter with correct core_map."""
        assert controller.rf_converter is not None
        assert 0 in controller.rf_converter.core_map
        assert 1 in controller.rf_converter.core_map

    def test_controller_build_target_mask(self, controller):
        mask = controller._build_target_mask((8, 8))
        assert mask.shape == (8, 8)
        assert mask.sum() == 16  # 4×4 target

    def test_controller_acquire_dummy(self, controller):
        img = controller._acquire()
        assert isinstance(img, np.ndarray)
        assert img.ndim == 2
        assert img.shape == (512, 512)

    def test_controller_output_batch_simulation(self, controller):
        """_output_batch in sim mode should not raise."""
        holding = controller.rf_converter.holding_config()
        controller._output_batch(holding)  # should log, not crash

    def test_controller_send_holding(self, controller):
        """_send_holding in sim mode should not raise."""
        controller._send_holding()

    def test_controller_rf_converter_holding_matches_grid(self, controller):
        """Holding config ramps should match grid_rows + grid_cols."""
        batch = controller.rf_converter.holding_config()
        s = controller.sw.aod_settings
        assert len(batch.ramps) == s.grid_rows + s.grid_cols


# =====================================================================
# 6. AODSettings validation tests
# =====================================================================

class TestAODSettings:

    def test_frequency_spacing_single_site(self):
        """Grid of 1 shouldn't divide by zero."""
        s = AODSettings(grid_rows=1, grid_cols=1)
        assert s.f_spacing_v == s.f_max_v - s.f_min_v
        assert s.f_spacing_h == s.f_max_h - s.f_min_h

    def test_validate_core_limits_valid(self):
        s = AODSettings(grid_rows=10, grid_cols=5)
        s.validate_core_limits()  # should not raise

    def test_validate_core_limits_invalid(self):
        s = AODSettings(grid_rows=10, grid_cols=6)
        with pytest.raises(ValueError):
            s.validate_core_limits()


# =====================================================================
# 7. RFRamp / AWGBatch data class tests
# =====================================================================

class TestDataClasses:

    def test_rf_ramp_defaults(self):
        ramp = RFRamp(channel=0, core=5, f_start=60e6, f_end=70e6, amplitude_pct=4.0)
        assert ramp.phase_deg == 0.0
        assert ramp.duration_s == 0.0

    def test_awg_batch_construction(self):
        ramps = [
            RFRamp(channel=0, core=i, f_start=60e6, f_end=60e6, amplitude_pct=4.0)
            for i in range(10)
        ]
        batch = AWGBatch(ramps=ramps, total_duration_s=0.0)
        assert len(batch.ramps) == 10
        assert batch.total_duration_s == 0.0


# =====================================================================
# 8. Edge-case tests
# =====================================================================

class TestEdgeCases:

    def test_identity_move_produces_static_ramp(self):
        """A move to the same position should produce f_start == f_end."""
        settings = _make_simple_settings(grid_rows=4, grid_cols=3)
        conv = RFConverter(settings, PhysicalParams())
        move = Move(from_row=2, from_col=1, to_row=2, to_col=1)
        batch = conv.convert_moves([move])
        for ramp in batch.ramps:
            assert ramp.f_start == ramp.f_end

    def test_multiple_parallel_moves(self):
        """Multiple non-conflicting parallel moves in one batch."""
        settings = _make_simple_settings(grid_rows=6, grid_cols=5)
        conv = RFConverter(settings, PhysicalParams())
        moves = [
            Move(from_row=0, from_col=0, to_row=1, to_col=1),
            Move(from_row=2, from_col=2, to_row=3, to_col=3),
            Move(from_row=4, from_col=4, to_row=5, to_col=4),
        ]
        batch = conv.convert_moves(moves)
        assert len(batch.ramps) == 6 + 5
        assert batch.total_duration_s > 0

    def test_same_row_different_col_targets_ok(self):
        """Moves from different rows but sharing a target col is legal
        as long as source cols have consistent targets."""
        settings = _make_simple_settings(grid_rows=4, grid_cols=4)
        conv = RFConverter(settings, PhysicalParams())
        moves = [
            Move(from_row=0, from_col=0, to_row=2, to_col=1),
            Move(from_row=1, from_col=0, to_row=3, to_col=1),
        ]
        # Both map col 0 → col 1, which is consistent
        batch = conv.convert_moves(moves)
        assert len(batch.ramps) == 4 + 4

    def test_holding_then_move_then_holding(self):
        """Simulate a round: holding → move batch → holding."""
        settings = _make_simple_settings(grid_rows=4, grid_cols=3)
        conv = RFConverter(settings, PhysicalParams())

        h1 = conv.holding_config()
        move_batch = conv.convert_moves([Move(0, 0, 1, 1)])
        h2 = conv.holding_config()

        # All holding ramps must be static
        for r in h1.ramps:
            assert r.f_start == r.f_end
        for r in h2.ramps:
            assert r.f_start == r.f_end


# =====================================================================
# 9. DDS strategy interface tests
# =====================================================================

class TestDDSStrategyInterface:
    """Verify that all strategies implement the required interface."""

    @pytest.fixture(params=list(STRATEGY_REGISTRY.keys()))
    def strategy(self, request) -> DDSStrategy:
        name = request.param
        if name == "camera_triggered":
            return get_strategy(name, trigger_level_v=1.0)
        return get_strategy(name)

    def test_strategy_has_name(self, strategy):
        assert isinstance(strategy.name, str)
        assert len(strategy.name) > 0

    def test_strategy_in_registry(self, strategy):
        assert strategy.name in STRATEGY_REGISTRY

    def test_strategy_has_required_methods(self, strategy):
        for method in [
            "create_dds", "configure", "prefill", "start",
            "execute_batch", "send_holding", "finalize_sequence",
            "shutdown",
        ]:
            assert hasattr(strategy, method), f"Missing method: {method}"
            assert callable(getattr(strategy, method))

    def test_compute_slope_static(self, strategy):
        """Static ramp (no motion) should yield zero slope."""
        ramp = RFRamp(channel=0, core=0, f_start=80e6, f_end=80e6,
                       amplitude_pct=4.0, duration_s=0.1)
        assert strategy.compute_slope(ramp) == 0.0

    def test_compute_slope_zero_duration(self, strategy):
        ramp = RFRamp(channel=0, core=0, f_start=60e6, f_end=80e6,
                       amplitude_pct=4.0, duration_s=0.0)
        assert strategy.compute_slope(ramp) == 0.0

    def test_compute_slope_positive(self, strategy):
        ramp = RFRamp(channel=0, core=0, f_start=60e6, f_end=80e6,
                       amplitude_pct=4.0, duration_s=0.5)
        slope = strategy.compute_slope(ramp)
        assert slope == pytest.approx(40e6, rel=1e-9)


class TestGetStrategy:
    """Verify the strategy factory function."""

    def test_valid_names(self):
        for name in STRATEGY_REGISTRY:
            if name == "camera_triggered":
                s = get_strategy(name, trigger_level_v=1.0)
            else:
                s = get_strategy(name)
            assert s.name == name

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown DDS strategy"):
            get_strategy("nonexistent")

    def test_ramp_with_kwargs(self):
        s = get_strategy("ramp", ramp_stepsize=500, use_scurve=True)
        assert isinstance(s, DDSRampStrategy)
        assert s.config.ramp_stepsize == 500
        assert s.config.use_scurve is True

    def test_pattern_with_kwargs(self):
        s = get_strategy("pattern", poll_interval_s=0.01, poll_timeout_s=5.0)
        assert isinstance(s, DDSPatternStrategy)
        assert s.config.poll_interval_s == pytest.approx(0.01)

    def test_camera_with_kwargs(self):
        s = get_strategy("camera_triggered", trigger_level_v=1.0)
        assert isinstance(s, DDSCameraTriggeredStrategy)
        assert s.config.trigger_level_v == pytest.approx(1.0)


# =====================================================================
# 10. DDSRampStrategy tests
# =====================================================================

class TestDDSRampStrategy:
    """Verify ramp slope computation and S-curve generation."""

    @pytest.fixture
    def strategy(self) -> DDSRampStrategy:
        return DDSRampStrategy()

    @pytest.fixture
    def scurve_strategy(self) -> DDSRampStrategy:
        return DDSRampStrategy(config=RampConfig(use_scurve=True, scurve_segments=8))

    def test_slope_for_timer_positive_ramp(self, strategy):
        ramp = RFRamp(channel=0, core=0, f_start=60e6, f_end=80e6,
                       amplitude_pct=4.0, duration_s=0.2)
        slope = strategy._slope_for_timer(ramp, timer_s=0.2)
        assert slope == pytest.approx(100e6, rel=1e-9)  # 20 MHz / 0.2s

    def test_slope_for_timer_negative_ramp(self, strategy):
        ramp = RFRamp(channel=0, core=0, f_start=80e6, f_end=60e6,
                       amplitude_pct=4.0, duration_s=0.2)
        slope = strategy._slope_for_timer(ramp, timer_s=0.2)
        assert slope == pytest.approx(-100e6, rel=1e-9)

    def test_slope_for_timer_no_motion(self, strategy):
        ramp = RFRamp(channel=0, core=0, f_start=70e6, f_end=70e6,
                       amplitude_pct=4.0, duration_s=0.2)
        slope = strategy._slope_for_timer(ramp, timer_s=0.2)
        assert slope == 0.0

    def test_scurve_slopes_sum_to_delta_f(self, scurve_strategy):
        """Sum of S-curve segment slopes × segment_time ≈ delta_f."""
        ramp = RFRamp(channel=0, core=0, f_start=60e6, f_end=80e6,
                       amplitude_pct=4.0, duration_s=0.2)
        slopes = scurve_strategy._compute_scurve_slopes(ramp, n_segments=8)
        assert len(slopes) == 8

        # Each segment lasts duration / n_segments
        dt = ramp.duration_s / 8
        total_delta = sum(s * dt for s in slopes)
        # Should approximately equal delta_f = 20 MHz
        assert total_delta == pytest.approx(20e6, rel=0.05)

    def test_scurve_slopes_symmetric(self, scurve_strategy):
        """S-curve slopes should be symmetric (acceleration == deceleration)."""
        ramp = RFRamp(channel=0, core=0, f_start=60e6, f_end=80e6,
                       amplitude_pct=4.0, duration_s=0.2)
        slopes = scurve_strategy._compute_scurve_slopes(ramp, n_segments=16)
        # First half slopes should mirror second half (reversed)
        for i in range(8):
            assert abs(slopes[i] - slopes[15 - i]) < 1e3, (
                f"Asymmetry at segment {i}: {slopes[i]} vs {slopes[15 - i]}"
            )

    def test_scurve_slopes_bell_shaped(self, scurve_strategy):
        """Peak slope should be in the middle segments (bell curve)."""
        ramp = RFRamp(channel=0, core=0, f_start=60e6, f_end=80e6,
                       amplitude_pct=4.0, duration_s=0.2)
        slopes = scurve_strategy._compute_scurve_slopes(ramp, n_segments=16)
        # Middle slopes should be larger than edge slopes
        mid_slope = slopes[8]
        edge_slope = slopes[0]
        assert abs(mid_slope) > abs(edge_slope)

    def test_scurve_static_ramp_all_zero(self, scurve_strategy):
        ramp = RFRamp(channel=0, core=0, f_start=70e6, f_end=70e6,
                       amplitude_pct=4.0, duration_s=0.2)
        slopes = scurve_strategy._compute_scurve_slopes(ramp, n_segments=8)
        assert all(s == 0.0 for s in slopes)

    def test_config_defaults(self):
        cfg = RampConfig()
        assert cfg.ramp_stepsize == 1000
        assert cfg.use_scurve is False
        assert cfg.scurve_segments == 16

    def test_strategy_name(self, strategy):
        assert strategy.name == "ramp"

    def test_trigger_timer_stored(self, strategy):
        """_trigger_timer_s should be initialized with a default."""
        assert strategy._trigger_timer_s > 0


# =====================================================================
# 11. DDSPatternStrategy tests
# =====================================================================

class TestDDSPatternStrategy:
    """Verify pattern strategy construction and configuration."""

    @pytest.fixture
    def strategy(self) -> DDSPatternStrategy:
        return DDSPatternStrategy()

    def test_strategy_name(self, strategy):
        assert strategy.name == "pattern"

    def test_config_defaults(self):
        cfg = PatternConfig()
        assert cfg.poll_interval_s == pytest.approx(0.001)
        assert cfg.poll_timeout_s == pytest.approx(10.0)

    def test_custom_config(self):
        cfg = PatternConfig(poll_interval_s=0.01, poll_timeout_s=5.0)
        s = DDSPatternStrategy(config=cfg)
        assert s.config.poll_interval_s == pytest.approx(0.01)
        assert s.config.poll_timeout_s == pytest.approx(5.0)

    def test_trigger_starts_none(self, strategy):
        """Internal trigger handle should be None before configure()."""
        assert strategy._trigger is None

    def test_shutdown_clears_trigger(self, strategy):
        strategy._trigger = "dummy"
        strategy.shutdown(dds=None)
        assert strategy._trigger is None


# =====================================================================
# 12. DDSCameraTriggeredStrategy tests
# =====================================================================

class TestDDSCameraTriggeredStrategy:
    """Verify camera trigger safety constraints and configuration."""

    def test_safe_trigger_level_accepted(self):
        """Trigger level < 2.0 V should be accepted."""
        s = DDSCameraTriggeredStrategy(
            config=CameraTriggerConfig(trigger_level_v=1.5)
        )
        assert s.config.trigger_level_v == pytest.approx(1.5)

    def test_unsafe_trigger_level_at_limit_raises(self):
        """Trigger level == 2.0 V should raise."""
        with pytest.raises(ValueError, match="safety limit"):
            DDSCameraTriggeredStrategy(
                config=CameraTriggerConfig(trigger_level_v=2.0)
            )

    def test_unsafe_trigger_level_above_limit_raises(self):
        """Trigger level > 2.0 V should raise."""
        with pytest.raises(ValueError, match="safety limit"):
            DDSCameraTriggeredStrategy(
                config=CameraTriggerConfig(trigger_level_v=3.0)
            )

    def test_unsafe_trigger_level_much_above_raises(self):
        """Trigger level = 5.0 V should raise."""
        with pytest.raises(ValueError, match="safety limit"):
            DDSCameraTriggeredStrategy(
                config=CameraTriggerConfig(trigger_level_v=5.0)
            )

    def test_minimum_trigger_level_accepted(self):
        """Very low trigger level should be accepted."""
        s = DDSCameraTriggeredStrategy(
            config=CameraTriggerConfig(trigger_level_v=0.1)
        )
        assert s.config.trigger_level_v == pytest.approx(0.1)

    def test_strategy_name(self):
        s = DDSCameraTriggeredStrategy(
            config=CameraTriggerConfig(trigger_level_v=1.0)
        )
        assert s.name == "camera_triggered"

    def test_config_defaults(self):
        cfg = CameraTriggerConfig()
        assert cfg.trigger_level_v == pytest.approx(1.5)
        assert cfg.trigger_coupling == "DC"
        assert cfg.trigger_edge == "rising"
        assert cfg.trigger_termination_ohms == pytest.approx(50.0)
        assert cfg.poll_timeout_s == pytest.approx(30.0)

    def test_falling_edge_config(self):
        s = DDSCameraTriggeredStrategy(
            config=CameraTriggerConfig(
                trigger_level_v=1.0,
                trigger_edge="falling",
            )
        )
        assert s.config.trigger_edge == "falling"

    def test_ac_coupling_config(self):
        s = DDSCameraTriggeredStrategy(
            config=CameraTriggerConfig(
                trigger_level_v=1.0,
                trigger_coupling="AC",
            )
        )
        assert s.config.trigger_coupling == "AC"

    def test_shutdown_clears_trigger(self):
        s = DDSCameraTriggeredStrategy(
            config=CameraTriggerConfig(trigger_level_v=1.0)
        )
        s._trigger = "dummy"
        s.shutdown(dds=None)
        assert s._trigger is None

    def test_max_safe_trigger_constant(self):
        """The safety constant should be 2.0 V."""
        assert MAX_SAFE_TRIGGER_LEVEL_V == pytest.approx(2.0)


# =====================================================================
# 13. DDSStreamingStrategy tests
# =====================================================================

class TestDDSStreamingStrategy:
    """Verify the streaming strategy (baseline comparison)."""

    @pytest.fixture
    def strategy(self) -> DDSStreamingStrategy:
        return DDSStreamingStrategy()

    def test_strategy_name(self, strategy):
        assert strategy.name == "streaming"

    def test_group_ramps_by_channel(self):
        """Helper function should group ramps correctly."""
        ramps = [
            RFRamp(channel=0, core=0, f_start=60e6, f_end=70e6, amplitude_pct=4.0),
            RFRamp(channel=1, core=20, f_start=60e6, f_end=70e6, amplitude_pct=8.0),
            RFRamp(channel=0, core=1, f_start=65e6, f_end=75e6, amplitude_pct=4.0),
        ]
        batch = AWGBatch(ramps=ramps, total_duration_s=0.1)
        groups = _group_ramps_by_channel(batch)
        assert 0 in groups
        assert 1 in groups
        assert len(groups[0]) == 2  # two ch0 ramps
        assert len(groups[1]) == 1  # one ch1 ramp


# =====================================================================
# 14. Strategy integration with controller (simulation mode)
# =====================================================================

class TestStrategyIntegration:
    """Test that strategies integrate with atommovrController."""

    @pytest.fixture(params=["streaming", "ramp", "pattern", "camera_triggered"])
    def controller_with_strategy(self, request):
        """Create a controller in sim mode with each strategy."""
        sys.path.insert(0, _SCRIPTS_DIR)
        from atommovr_controller import (
            atommovrController,
            HardwareConfig,
            SoftwareConfig,
        )

        strategy_name = request.param
        if strategy_name == "camera_triggered":
            strategy = get_strategy(strategy_name, trigger_level_v=1.0)
        else:
            strategy = get_strategy(strategy_name)

        sw = SoftwareConfig(
            grid_size=8,
            target_size=4,
            algorithm_name="PCFA",
            max_rounds=3,
            aod_settings=_make_simple_settings(
                grid_rows=8, grid_cols=5, target_rows=4, target_cols=4,
            ),
        )
        hw = HardwareConfig()
        ctrl = atommovrController(sw, hw, strategy=strategy)
        yield ctrl
        ctrl.shutdown()

    def test_strategy_assigned(self, controller_with_strategy):
        ctrl = controller_with_strategy
        assert isinstance(ctrl.strategy, DDSStrategy)
        assert ctrl.strategy.name in STRATEGY_REGISTRY

    def test_rf_converter_initialised(self, controller_with_strategy):
        ctrl = controller_with_strategy
        assert ctrl.rf_converter is not None
        assert 0 in ctrl.rf_converter.core_map
        assert 1 in ctrl.rf_converter.core_map

    def test_output_batch_sim_mode(self, controller_with_strategy):
        """_output_batch in sim mode should not raise for any strategy."""
        ctrl = controller_with_strategy
        holding = ctrl.rf_converter.holding_config()
        ctrl._output_batch(holding)  # should log, not crash

    def test_send_holding_sim_mode(self, controller_with_strategy):
        """_send_holding in sim mode should not raise for any strategy."""
        ctrl = controller_with_strategy
        ctrl._send_holding()

    def test_controller_string_strategy(self):
        """Controller should accept strategy as a string name."""
        sys.path.insert(0, _SCRIPTS_DIR)
        from atommovr_controller import (
            atommovrController,
            HardwareConfig,
            SoftwareConfig,
        )

        sw = SoftwareConfig(
            grid_size=6,
            target_size=4,
            algorithm_name="PCFA",
            aod_settings=_make_simple_settings(grid_rows=6, grid_cols=4),
        )
        hw = HardwareConfig()
        ctrl = atommovrController(sw, hw, strategy="ramp")
        assert ctrl.strategy.name == "ramp"
        ctrl.shutdown()

    def test_controller_default_strategy(self):
        """Default strategy should be streaming."""
        sys.path.insert(0, _SCRIPTS_DIR)
        from atommovr_controller import (
            atommovrController,
            HardwareConfig,
            SoftwareConfig,
        )

        sw = SoftwareConfig(
            grid_size=6,
            target_size=4,
            algorithm_name="PCFA",
            aod_settings=_make_simple_settings(grid_rows=6, grid_cols=4),
        )
        hw = HardwareConfig()
        ctrl = atommovrController(sw, hw)
        assert ctrl.strategy.name == "streaming"
        ctrl.shutdown()
