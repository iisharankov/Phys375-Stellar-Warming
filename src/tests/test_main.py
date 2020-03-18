import pytest

from src import const, main

# @pytest.fixture(scope="module")
# def mt():
#     star = main.Star()
#     return (star)

class TestMainSequence:
    def test_x_y_z_vals(self):
        star = main.Star(0.5, 0.25, 0.25)
        assert star.x + star.y + star.z == 1
        assert star.luminosity == const.L_SUN
        assert star.radius == const.R_SUN
        assert star.mass == const.M_SUN

        star = main.Star(0.5, 0.25, 0.25, 1, 1, 1)
        assert star.x + star.y + star.z == 1
        assert star.luminosity == const.L_SUN
        assert star.radius == const.R_SUN
        assert star.mass == const.M_SUN

        star = main.Star(0.5, 0.25, 0.25, 2, 4, 6)
        assert star.x + star.y + star.z == 1
        assert star.mass == const.M_SUN * 2
        assert star.radius == const.R_SUN * 4
        assert star.luminosity == const.L_SUN * 6
