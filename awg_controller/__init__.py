"""AWG Controller - Spectrum Instrumentation DDS card driver for atom rearrangement.

This package isolates all AWG (Arbitrary Waveform Generator) and DDS
(Direct Digital Synthesis) control logic from the main ``atommovr``
simulation framework.

Subpackages
-----------
src/
    Core modules: ``awg_control`` (RF converter, hardware constants),
    ``dds_strategies`` (four interchangeable DDS execution strategies).
docs/
    Strategy-specific documentation with safety instructions.
tests/
    Unit and integration tests for all AWG modules.
config/
    Hardware configuration files and templates.
scripts/
    ``atommovr_controller.py`` - closed-loop rearrangement controller.
"""
