"""AWG controller source modules.

Re-exports the public API from ``awg_control`` and ``dds_strategies``
so users can write::

    from awg_controller.src import RFConverter, AWGBatch, DDSRampStrategy
"""

from awg_controller.src.awg_control import (
    AODSettings,
    AWGBatch,
    RFConverter,
    RFRamp,
    compute_core_assignments,
    validate_hardware_limits,
    MAX_AMPLITUDE_PCT_PER_CHANNEL,
    TOTAL_DDS_CORES,
)
from awg_controller.src.dds_strategies import (
    DDSStrategy,
    DDSStreamingStrategy,
    DDSRampStrategy,
    DDSPatternStrategy,
    DDSCameraTriggeredStrategy,
    RampConfig,
    PatternConfig,
    CameraTriggerConfig,
    STRATEGY_REGISTRY,
    get_strategy,
)

__all__ = [
    # awg_control
    "AODSettings",
    "AWGBatch",
    "RFConverter",
    "RFRamp",
    "compute_core_assignments",
    "validate_hardware_limits",
    "MAX_AMPLITUDE_PCT_PER_CHANNEL",
    "TOTAL_DDS_CORES",
    # dds_strategies
    "DDSStrategy",
    "DDSStreamingStrategy",
    "DDSRampStrategy",
    "DDSPatternStrategy",
    "DDSCameraTriggeredStrategy",
    "RampConfig",
    "PatternConfig",
    "CameraTriggerConfig",
    "STRATEGY_REGISTRY",
    "get_strategy",
]
