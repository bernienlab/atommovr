import numpy as np

from atommovr.utils.imaging.animation import (
    _get_inds_for_circ_matr_plot,
    _dual_species_get_inds_for_circ_matr_plot,
    _check_and_fix_lims,
)


class TestGetIndsForCircMatrPlot:
    def test_basic_matrix(self):
        matrix = np.array([[1, 0], [0, 1]])
        filled_x, filled_y, empty_x, empty_y = _get_inds_for_circ_matr_plot(matrix)
        assert set(zip(filled_x, filled_y, strict=True)) == {(0, 0), (1, 1)}
        assert set(zip(empty_x, empty_y, strict=True)) == {(1, 0), (0, 1)}

    def test_all_filled(self):
        matrix = np.ones((3, 3))
        filled_x, filled_y, empty_x, empty_y = _get_inds_for_circ_matr_plot(matrix)
        assert len(filled_x) == 9
        assert len(empty_x) == 0
        assert len(filled_y) == 9
        assert len(empty_y) == 0

    def test_all_empty(self):
        matrix = np.zeros((3, 3))
        filled_x, filled_y, empty_x, empty_y = _get_inds_for_circ_matr_plot(matrix)
        assert len(filled_x) == 0
        assert len(empty_x) == 9
        assert len(filled_y) == 0
        assert len(empty_y) == 9

    def test_3d_matrix(self):
        matrix = np.array([[[1], [0]], [[0], [1]]])
        filled_x, filled_y, empty_x, empty_y = _get_inds_for_circ_matr_plot(matrix)
        assert len(filled_x) == 2
        assert len(empty_x) == 2
        assert len(filled_y) == 2
        assert len(empty_y) == 2


class TestDualSpeciesGetInds:
    def test_basic_dual_species(self):
        matrix = np.zeros((2, 2, 2))
        matrix[0, 0, 0] = 1  # Species 0 at (0,0)
        matrix[1, 1, 1] = 1  # Species 1 at (1,1)
        blue_x, blue_y, yellow_x, yellow_y, white_x, white_y = (
            _dual_species_get_inds_for_circ_matr_plot(matrix)
        )
        assert len(blue_x) == 1
        assert len(yellow_x) == 1
        assert len(white_x) == 2
        assert len(blue_y) == 1
        assert len(yellow_y) == 1
        assert len(white_y) == 2

    def test_all_species_0(self):
        matrix = np.zeros((2, 2, 2))
        matrix[:, :, 0] = 1
        blue_x, blue_y, yellow_x, yellow_y, white_x, white_y = (
            _dual_species_get_inds_for_circ_matr_plot(matrix)
        )
        assert len(blue_x) == 4
        assert len(yellow_x) == 0
        assert len(white_x) == 0

    def test_empty_matrix(self):
        matrix = np.zeros((3, 3, 2))
        blue_x, blue_y, yellow_x, yellow_y, white_x, white_y = (
            _dual_species_get_inds_for_circ_matr_plot(matrix)
        )
        assert len(blue_x) == 0
        assert len(yellow_x) == 0
        assert len(white_x) == 9


class TestCheckAndFixLims:
    def test_expands_limits_when_needed(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 5])
        _check_and_fix_lims(ax, 10, 10)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim[1] >= 10
        assert ylim[1] >= 10
        plt.close()

    def test_keeps_limits_when_sufficient(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_xlim([0, 20])
        ax.set_ylim([0, 20])
        _check_and_fix_lims(ax, 5, 5)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim[1] == 20
        assert ylim[1] == 20
        plt.close()
