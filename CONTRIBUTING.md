# Contributing to atommovr

Interested in building out *atommovr* or adding a custom feature? You've come to the right place. 

*atommovr* was intentionally designed to be customizable to every user's needs. Instead of being sucked into the black hole of trying to code up *every imaginable feature and use case* (and inevitably failing), we tried to keep *atommovr* very modular so that users could easily add and/or amend features. Because after all, who knows your use case better than *you*?

In particular, the `Algorithm` and `ErrorModel` classes are very general, and their source files describe the requirements for implementing new algorithms or error models.

## Testing requirements for PRs
If a PR fixes a bug, please add a regression test that would have failed before the fix.
A regression test is a test that protects against a specific bug returning during future refactors.
Recommended naming pattern:
def test_regression_<short_bug_description>():
    ...
Examples:
- seeded RNG mismatch in a helper function
- missing super().__init__() in a child error model
- incorrect return value from a loss function

## Specific opportunities for contribution

*Code maintenance and performance*
- Adding more unit tests to `atommovr.tests/`
- Speeding up the core functions in `atommovr.utils.move_utils.py`, `atommovr.utils.core.py`. (This code is not currently optimized for speed.)
- Speeding up the gif generation process in `atommovr.utils.animation.py` (currently very slow).

*New features*
- Extending framework to support general array/lattice shapes
  - i.e. by building out the `ArrayGeometry` class in `atommovr.utils.core.py` and building support for animations in `atommovr.utils.animation.py`
- Adding automatic plotting to the benchmarking module
  - i.e. by building out the `BenchmarkingFigure` class in `atommovr.utils.benchmarking.py`

*Growing the library*
- Adding more algorithms (see `atommovr.algorithms.Algorithm.py` for a template)
- Adding more error models (see `atommovr.utils.ErrorModel.py` for a template)

Thinking of something that's not on this list? Feel free to contact [Nikhil](mailto:nikhil.calvin@gmail.com).

## Conventions for dtypes
To avoid subtle bugs and dtype drift, please follow these conventions:
- `AtomArray.matrix`: `np.uint8` (occupancy arrays)
- per-move event_mask: `np.uint64` (bitmask of `FailureBit`s)
- eligibility masks: `bool` arrays (`shape == event_mask.shape`)
- resolved primary events: integer code array (currently `np.int32`)