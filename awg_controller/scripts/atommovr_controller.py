#!/usr/bin/env python3
"""Compatibility shim for tests/scripts importing `atommovr_controller`.

The production controller implementation lives in `atommovr_controller.py`.
This module re-exports its public symbols to preserve stable import paths.
"""

from atommovr_controller import *  # noqa: F401,F403
