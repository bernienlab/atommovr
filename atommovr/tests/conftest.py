"""
Shared pytest fixtures for the test suite.

Only place broadly reused, pytest-specific setup here. Keep pure helper
functions in `tests.support.helpers` and test doubles in
`tests.support.doubles`.
"""

from __future__ import annotations

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")

from atommovr.utils.AtomArray import AtomArray
from atommovr.tests.support.doubles import TimingSpyErrorModel


@pytest.fixture
def rng() -> np.random.Generator:
    """
    Return a deterministic RNG for reproducible tests.
    """
    return np.random.default_rng(0)


@pytest.fixture
def timing_error_model() -> TimingSpyErrorModel:
    """
    Return a deterministic spy error model for seam/orchestration tests.
    """
    return TimingSpyErrorModel()


@pytest.fixture
def empty_3x3_atomarray(timing_error_model: TimingSpyErrorModel) -> AtomArray:
    """
    Return a zero-filled single-species AtomArray for basic move tests.
    """
    aa = AtomArray(shape=[3, 3], n_species=1, error_model=timing_error_model)
    aa.matrix[:, :, 0] = np.uint8(0)
    return aa


@pytest.fixture
def one_atom_array_3x3(timing_error_model: TimingSpyErrorModel) -> AtomArray:
    """
    Return a small AtomArray with a single atom at the origin.
    """
    aa = AtomArray(shape=[3, 3], n_species=1, error_model=timing_error_model)
    aa.matrix[:, :, 0] = np.uint8(0)
    aa.matrix[0, 0, 0] = np.uint8(1)
    return aa
