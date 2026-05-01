#!/usr/bin/env python3
"""
atommovr Production Controller
================================
Orchestrates the complete atom-rearrangement feedback loop:

  1. Acquire fluorescence image (file or camera callback)
  2. Blob detection → rotation correction → grid assignment
  3. Check whether target is already filled  →  done
  4. Compute rearrangement moves (configurable algorithm)
  5. Convert moves to RF ramps  (RFConverter / AWGBatch)
  6. Write ramps to Spectrum Instrumentation DDS card  (spcm)
  7. Wait for physical move to complete
  8. Repeat from step 1

Hardware integration is copied verbatim from cli.py (source of truth):
  - DDSCommandQueue for all core updates
  - dds[core].freq() / .phase() / .amp()  in spcm units
  - exec_at_trg() + write_to_card() per trigger event
  - spcm.SPCM_DDS_TRG_SRC_TIMER trigger mode
  - 40 % total amplitude budget per channel
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

#  repo root on sys.path 
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

#  optional hardware driver 
try:
    import spcm
    from spcm import SpcmException
    _HW_AVAILABLE = True
except ImportError:
    spcm = None                 # type: ignore[assignment]
    SpcmException = Exception   # type: ignore[assignment,misc]
    _HW_AVAILABLE = False

#  atommovr imports 
from atommovr.algorithms.single_species import (
    PCFA, Hungarian, Tetris, BalanceAndCompact,
    BCv2, ParallelLBAP, ParallelHungarian, GeneralizedBalance,
)
from atommovr.utils.AtomArray import AtomArray
from awg_controller.src.awg_control import (
    AODSettings, AWGBatch,
    MAX_AMPLITUDE_PCT_PER_CHANNEL, RFConverter, RFRamp,
    validate_hardware_limits,
)
from atommovr.utils.core import Configurations, PhysicalParams
from awg_controller.src.dds_strategies import (
    DDSStrategy,
    DDSStreamingStrategy,
    STRATEGY_REGISTRY,
    get_strategy,
)
from atommovr.utils.imaging.extraction import (
    BlobDetection,
    estimate_grid_rotation_fit_rect,
    fit_grid_and_assign,
    inverse_rotate_centroids,
)

#  logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("atommovr_controller.log"),
    ],
)
log = logging.getLogger(__name__)

if not _HW_AVAILABLE:
    log.warning("spcm not found - running in SIMULATION mode (no hardware I/O).")


@dataclass
class HardwareConfig:
    """Spectrum Instrumentation card configuration (mirrors cli.py defaults)."""

    #: Device paths, e.g. ["/dev/spcm0"]  or  ["/dev/spcm0", "/dev/spcm1"]
    card_paths: List[str] = field(default_factory=lambda: ["/dev/spcm0"])

    #: Output amplitude - manufacturer maximum is 1.6 V into 50 Ω
    max_amplitude_v: float = 1.6

    #: Output impedance
    output_load_ohms: float = 50.0

    #: Timer-trigger interval (s).  Set to the expected move duration so the
    #: card fires once per rearrangement batch.  cli.py default was 0.2 s.
    trigger_timer_s: float = 0.2

    #: Number of pre-fill writes during initialisation (prevents underruns).
    #: cli.py uses  floor(10 / trg_timer); we keep that formula.
    @property
    def prefill_count(self) -> int:
        return max(1, math.floor(10.0 / self.trigger_timer_s))


@dataclass
class SoftwareConfig:
    """Algorithm and imaging configuration."""

    #  Imaging 
    #: Parameters forwarded to BlobDetection (calibrate once per setup).
    blob_params: Dict[str, Any] = field(default_factory=lambda: {
        "min_sigma": 1.0, "max_sigma": 5.0, "threshold": 0.05,
    })

    #  Trap geometry 
    #: Side length of the full trap grid (L × L sites total).
    grid_size: int = 10

    #: Side length of the target sub-array (square).
    target_size: int = 6

    #: AOD frequency-range and geometry.
    aod_settings: AODSettings = field(default_factory=AODSettings)

    #: Physical parameters (AOD speed, site spacing, loading probability …).
    physical_params: PhysicalParams = field(default_factory=PhysicalParams)

    #  Control loop 
    #: Maximum rearrangement rounds before giving up.
    max_rounds: int = 10

    #: Algorithm to use (class name as string, e.g. "PCFA").
    algorithm_name: str = "PCFA"

    #: Target pattern type (default: centred square filled block).
    target_type: Configurations = Configurations.MIDDLE_FILL



_ALGORITHM_REGISTRY = {
    "PCFA":              PCFA,
    "Hungarian":         Hungarian,
    "Tetris":            Tetris,
    "BalanceAndCompact": BalanceAndCompact,
    "BCv2":              BCv2,
    "ParallelLBAP":      ParallelLBAP,
    "ParallelHungarian": ParallelHungarian,
    "GeneralizedBalance":GeneralizedBalance,
}


class atommovrController:
    """End-to-end atom rearrangement controller.

    Parameters
    ----------
    sw_config : SoftwareConfig
        Algorithm, imaging and geometry settings.
    hw_config : HardwareConfig
        Spectrum Instrumentation card settings.
    camera_fn : callable, optional
        ``camera_fn() -> np.ndarray``
        Called on every acquisition after the first image.  If *None*, the
        controller uses the internal dummy generator (for simulation / testing).
    strategy : DDSStrategy or str, optional
        DDS execution strategy.  Accepts a :class:`DDSStrategy` instance or
        a string name from :data:`STRATEGY_REGISTRY`
        (``"streaming"``, ``"ramp"``, ``"pattern"``,
        ``"camera_triggered"``).  Defaults to ``DDSStreamingStrategy``.
    """

    def __init__(
        self,
        sw_config: SoftwareConfig,
        hw_config: HardwareConfig,
        camera_fn: Optional[Callable[[], np.ndarray]] = None,
        strategy: Optional[DDSStrategy | str] = None,
    ) -> None:
        self.sw   = sw_config
        self.hw   = hw_config
        self._cam = camera_fn

        # Resolve strategy
        if strategy is None:
            self.strategy: DDSStrategy = DDSStreamingStrategy()
        elif isinstance(strategy, str):
            self.strategy = get_strategy(strategy)
        else:
            self.strategy = strategy

        self.algorithm   = _ALGORITHM_REGISTRY[sw_config.algorithm_name]()
        self.rf_converter = RFConverter(sw_config.aod_settings, sw_config.physical_params)

        # per-card DDS objects  {card_sn: DDS or DDSCommandQueue}
        self._dds: Dict[int, Any] = {}
        self._stack: Any = None         # spcm.CardStack (or None in sim mode)

        self._grid_rotation: float = 0.0   # updated each image acquisition
        self._target_mask: Optional[np.ndarray] = None  # built on first image

        self._initialize_hardware()


    def _initialize_hardware(self) -> None:
        """Open cards, configure DDS, pre-fill buffer, enable trigger.

        Delegates strategy-specific logic (DDS type, trigger mode,
        pre-fill, start) to ``self.strategy``.

        Common setup order (invariant across strategies)::

            1. card_mode(SPC_REP_STD_DDS)
            2. Create Channels → enable, set amp, output load
            3. card.write_setup()  — activates clock signals
            4. strategy.create_dds(card, channels)
            5. strategy.configure(dds, card, hw_config, core_map)
            6. strategy.prefill(dds, holding_batch, hw_config)
            7. strategy.start(stack)
        """
        if not _HW_AVAILABLE:
            log.info(
                f"Simulation mode: hardware init skipped "
                f"(strategy={self.strategy.name})."
            )
            return

        # Validate grid dimensions against hardware core limits
        validate_hardware_limits(
            self.sw.aod_settings.grid_rows,
            self.sw.aod_settings.grid_cols,
        )

        try:
            self._stack = spcm.CardStack(self.hw.card_paths)
            holding = self.rf_converter.holding_config()
            core_map = self.rf_converter.core_map

            for card in self._stack.cards:
                sn = card.sn()

                # 1. Set DDS card mode
                card.card_mode(spcm.SPC_REP_STD_DDS)

                # 2. Enable both output channels (V-AOD + H-AOD)
                channels = spcm.Channels(
                    card=card,
                    card_enable=spcm.CHANNEL0 | spcm.CHANNEL1,
                )
                channels.enable(True)
                channels.amp(self.hw.max_amplitude_v * spcm.units.V)
                channels.output_load(self.hw.output_load_ohms * spcm.units.ohm)

                # 3. Activate card clock (MUST come after channel config)
                card.write_setup()

                # 4-6. Strategy-specific DDS creation + configuration
                dds = self.strategy.create_dds(card, channels)
                self.strategy.configure(dds, card, self.hw, core_map)
                self.strategy.prefill(dds, holding, self.hw)

                self._dds[sn] = dds

            # 7. Start (strategy decides flags)
            self.strategy.start(self._stack)
            log.info(
                f"Hardware initialised (strategy={self.strategy.name})."
            )

        except SpcmException as exc:
            log.error(f"spcm error during init: {exc}")
            self._close_stack()
            raise
        except Exception as exc:
            log.error(f"Unexpected error during init: {exc}")
            self._close_stack()
            raise

    def _output_batch(self, batch: AWGBatch) -> None:
        """Send one ``AWGBatch`` to all open cards via the active strategy.

        Simulation mode logs the batch parameters instead of touching hardware.
        """
        if not _HW_AVAILABLE or self._stack is None:
            log.info(
                f"[SIM:{self.strategy.name}] batch: {len(batch.ramps)} ramps, "
                f"duration={batch.total_duration_s*1e6:.1f} µs"
            )
            if batch.total_duration_s > 0:
                time.sleep(batch.total_duration_s)
            return

        for card in self._stack.cards:
            sn  = card.sn()
            dds = self._dds[sn]
            self.strategy.execute_batch(dds, batch)

    def _send_holding(self) -> None:
        """Restore static holding configuration (atoms held in place)."""
        holding = self.rf_converter.holding_config()

        if not _HW_AVAILABLE or self._stack is None:
            log.info(
                f"[SIM:{self.strategy.name}] holding: "
                f"{len(holding.ramps)} ramps"
            )
            return

        for card in self._stack.cards:
            sn  = card.sn()
            dds = self._dds[sn]
            self.strategy.send_holding(dds, holding)


    def _acquire(self, path: Optional[str] = None) -> np.ndarray:
        """Return a grayscale image array.

        Priority:
        1. ``path`` (first image from disk)
        2. ``self._cam()`` (real camera callback)
        3. Dummy random image (simulation / testing)
        """
        if path and Path(path).exists():
            import cv2  # soft-import: only needed for disk images
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise OSError(f"cv2 could not read image at '{path}'")
            return img

        if self._cam is not None:
            return self._cam()

        # Dummy: random fluorescence image for off-hardware testing
        rng = np.random.default_rng()
        grid = self.sw.grid_size
        img  = np.zeros((512, 512), dtype=np.uint8)
        # Sprinkle bright blobs at grid positions
        step = 512 // grid
        for r in range(grid):
            for c in range(grid):
                if rng.random() < self.sw.physical_params.loading_prob:
                    cy, cx = r * step + step // 2, c * step + step // 2
                    img[max(0, cy-2):cy+3, max(0, cx-2):cx+3] = 200
        log.debug("Dummy image generated (simulation mode).")
        return img


    def _process_image(self, image: np.ndarray) -> np.ndarray:
        """Blob detect → rotation correct → grid assign → binary matrix.

        Returns
        -------
        np.ndarray, shape (grid_size, grid_size), dtype int
            Binary occupancy matrix (1 = atom present, 0 = empty).
        """
        t0 = time.perf_counter()

        # 1. Blob detection
        blobs = BlobDetection(self.sw.blob_params).detect(image)
        if len(blobs) == 0:
            log.warning("No blobs detected - returning empty occupancy matrix.")
            return np.zeros((self.sw.grid_size, self.sw.grid_size), dtype=int)

        centroids = np.array([[b.x, b.y] for b in blobs], dtype=float)

        # 2. Rotation estimation (fit_rect per project conventions)
        rotation = estimate_grid_rotation_fit_rect(blobs)
        self._grid_rotation = rotation

        # 3. Inverse-rotate centroids into an axis-aligned frame
        if abs(rotation) > 0.01:
            centroids = inverse_rotate_centroids(
                centroids, image_shape=image.shape[:2], angle_deg=rotation
            )

        # 4. Assign centroids to the closest grid sites
        binary = fit_grid_and_assign(
            centroids,
            (self.sw.grid_size, self.sw.grid_size),
            image_shape=image.shape[:2],
        )

        dt_ms = (time.perf_counter() - t0) * 1e3
        log.debug(
            f"Image processed in {dt_ms:.1f} ms: "
            f"{int(binary.sum())} atoms on {self.sw.grid_size}×{self.sw.grid_size} grid, "
            f"rotation={rotation:.2f}°"
        )
        return binary

    def _build_target_mask(self, grid_shape: Tuple[int, int]) -> np.ndarray:
        """Create a centred square target mask of side ``target_size``."""
        rows, cols = grid_shape
        t = self.sw.target_size
        mask = np.zeros((rows, cols), dtype=int)
        r0 = max((rows - t) // 2, 0)
        c0 = max((cols - t) // 2, 0)
        t_r = min(t, rows - r0)
        t_c = min(t, cols - c0)
        mask[r0:r0 + t_r, c0:c0 + t_c] = 1
        return mask

    def run(self, initial_image: Optional[str] = None) -> bool:
        """Execute the rearrangement feedback loop.

        Parameters
        ----------
        initial_image : str, optional
            Path to a fluorescence image for the first acquisition round.
            Subsequent rounds use ``camera_fn`` or the dummy generator.

        Returns
        -------
        bool
            ``True`` if the target is successfully filled within
            ``max_rounds``, ``False`` otherwise.
        """
        log.info(
            f"Loop start — algorithm={self.sw.algorithm_name}, "
            f"target={self.sw.target_size}×{self.sw.target_size}, "
            f"max_rounds={self.sw.max_rounds}"
        )

        image_path = initial_image  # used only for round 0

        for r in range(self.sw.max_rounds + 1):
            t_loop = time.perf_counter()

            #  1. Acquire 
            try:
                img = self._acquire(image_path)
                image_path = None   # subsequent rounds use camera / dummy
            except Exception as exc:
                log.error(f"Round {r}: acquisition failed - {exc}")
                return False

            #  2. Process 
            state = self._process_image(img)

            #  3. Build target mask (once) 
            if self._target_mask is None:
                self._target_mask = self._build_target_mask(state.shape)

            #  4. Check success 
            target = self._target_mask
            if np.array_equal(state * target, target):
                log.info(f"SUCCESS - target filled after {r} round(s).")
                return True

            if r == self.sw.max_rounds:
                log.warning(
                    f"Max rounds ({self.sw.max_rounds}) reached; "
                    f"{int((target - state * target).clip(0).sum())} sites remain empty."
                )
                break

            #  5. Build AtomArray for algorithm 
            arr = AtomArray(list(state.shape), n_species=1,
                            params=self.sw.physical_params)
            arr.matrix[:, :, 0] = state
            arr.target[:, :, 0] = target

            if int(state.sum()) < int(target.sum()):
                log.error(
                    f"Round {r}: insufficient atoms "
                    f"(have {int(state.sum())}, need {int(target.sum())}). Aborting."
                )
                return False

            #  6. Compute moves 
            try:
                _, move_batches, algo_ok = self.algorithm.get_moves(arr)
            except Exception as exc:
                log.exception(f"Round {r}: algorithm raised - {exc}")
                return False

            if not algo_ok:
                log.error(f"Round {r}: algorithm reported failure.")
                return False

            #  7. Convert & drive hardware 
            rf_batches = self.rf_converter.convert_sequence(move_batches)
            log.info(f"Round {r}: {len(rf_batches)} hardware batches.")

            for b_idx, batch in enumerate(rf_batches):
                self._output_batch(batch)

            # Restore static holding config after the whole sequence
            self._send_holding()

            elapsed_ms = (time.perf_counter() - t_loop) * 1e3
            log.info(f"Round {r} done in {elapsed_ms:.1f} ms.")

        return False

    # Shutdown

    def _close_stack(self) -> None:
        if self._stack is not None:
            # Strategy-specific cleanup per card
            for card in self._stack.cards:
                sn = card.sn()
                dds = self._dds.get(sn)
                if dds is not None:
                    self.strategy.shutdown(dds, card)
            try:
                self._stack.close()
            except Exception:
                pass
            self._stack = None
            self._dds.clear()

    def shutdown(self) -> None:
        """Gracefully stop the card and release all resources."""
        self._close_stack()
        log.info(f"Controller shut down (strategy={self.strategy.name}).")

    #  context manager support 
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()


# CLI entry-point

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="atommovr production controller — image → AOD feedback loop."
    )
    p.add_argument("--image",      default=None,   help="Path to initial fluorescence image")
    p.add_argument("--algorithm",  default="PCFA",
                   choices=list(_ALGORITHM_REGISTRY), help="Rearrangement algorithm")
    p.add_argument("--grid-rows",  type=int, default=10, help="Grid rows (V-AOD tones, max 16 w/ 2ch)")
    p.add_argument("--grid-cols",  type=int, default=5,  help="Grid cols (H-AOD tones, max 5)")
    p.add_argument("--target-rows",type=int, default=6,  help="Target sub-array rows")
    p.add_argument("--target-cols",type=int, default=5,  help="Target sub-array cols")
    p.add_argument("--max-rounds", type=int, default=10, help="Max rearrangement rounds")
    p.add_argument("--card",       action="append", default=["/dev/spcm0"],
                   help="Card device path (repeat for multiple cards)")
    p.add_argument("--trg-timer",  type=float, default=0.2,
                   help="Trigger timer interval (s), sets move window")
    p.add_argument("--f-min-v",    type=float, default=60e6, help="V-AOD f_min (Hz)")
    p.add_argument("--f-max-v",    type=float, default=100e6, help="V-AOD f_max (Hz)")
    p.add_argument("--f-min-h",    type=float, default=60e6, help="H-AOD f_min (Hz)")
    p.add_argument("--f-max-h",    type=float, default=100e6, help="H-AOD f_max (Hz)")
    p.add_argument("--strategy",   default="streaming",
                   choices=list(STRATEGY_REGISTRY),
                   help="DDS execution strategy")
    args = p.parse_args()

    hw = HardwareConfig(
        card_paths=args.card,
        trigger_timer_s=args.trg_timer,
    )
    sw = SoftwareConfig(
        grid_size=max(args.grid_rows, args.grid_cols),
        target_size=min(args.target_rows, args.target_cols),
        algorithm_name=args.algorithm,
        max_rounds=args.max_rounds,
        aod_settings=AODSettings(
            f_min_v=args.f_min_v, f_max_v=args.f_max_v,
            f_min_h=args.f_min_h, f_max_h=args.f_max_h,
            grid_rows=args.grid_rows, grid_cols=args.grid_cols,
            target_rows=args.target_rows, target_cols=args.target_cols,
        ),
    )

    with atommovrController(sw, hw, strategy=args.strategy) as ctrl:
        try:
            success = ctrl.run(initial_image=args.image)
            sys.exit(0 if success else 1)
        except KeyboardInterrupt:
            log.info("Interrupted by user.")
            sys.exit(130)


if __name__ == "__main__":
    main()