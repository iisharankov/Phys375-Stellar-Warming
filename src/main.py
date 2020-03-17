from src import const

tesst = 5
class MainSequence:

    def __init__(self):
        self.temp = 2


class Star:

    def __init__(self, x, y, z, mass_ratio=1, radius_ratio=1, luminosity_ratio=1):
        self.x = x
        self.y = y
        self.z = z
        self.mass = mass_ratio * const.M_SUN
        self.radius = radius_ratio * const.R_SUN
        self.luminosity = luminosity_ratio * const.L_SUN

        self.check_mass_fractions()

    def check_mass_fractions(self):
        """
        Makes sure the total value of the X, Y, Z mass fractions is 1
        """
        try:
            assert 0 <= self.x <= 1
            assert 0 <= self.y <= 1
            assert 0 <= self.z <= 1
            assert self.x + self.y + self.z == 1
        except AssertionError as E:
            print("Mass fractions did not add up to one!")
            raise E



