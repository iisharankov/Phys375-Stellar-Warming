import numpy as np
import time
import matplotlib.pyplot as plt
# from src.Star_Creator import SingleStar

# Tries to import multiprocessing, if failed, MP_BOOL is False for later use
try:
    import multiprocessing as multiproc
    MP_BOOL = True
except:
    MP_BOOL = False

from numba import jit
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import time

# General Constants
MASS_E = 9.10938291e-31  # electron mass
MASS_P = 1.67262178e-27  # proton mass
STEF_BOLT = 5.670373e-8  # Stefan-Boltzmann constant
K_BOLT = 1.381e-23  # Boltzmann constant
C = 299792458  # speed of light
G = 6.673e-11  # Gravitation constant
HBAR = 1.054571817e-34

# Astronomy constants
RAD_CONST = (4 * STEF_BOLT) / C  # radiation constant
X = 0.73  # Hydrogen mass fraction
Y = 0.25  # Helium mass fraction
Z = 0.02  # Metals mass fraction)k
# GAMMA = 1.9999  # adiabatic constant
M_SUN = 1.989e30  # Kg
R_SUN = 6.955e8  # meters
L_SUN = 3.827e26  # watts
ION_GAS_APPROX = (2 * X + 0.75 * Y + 0.5 * Z) ** -1  # approximation for ionized gas? u = 2X + 0.75Y + 0.5Z
KAPPA = 1
K = 1.38e-23
pi = np.pi

mu = (2.0 * X + 0.75 * Y + 0.5 * Z) ** (-1.0)
r0 = 0.001  # m
S = 1.0  # error tolerance


# Pressures
@jit(nopython=True)
def P(density, temp):
    # Pressure degenerate
    # Eqn 5 in the project_description
    P_deg = (pow((3.0 * pi ** 2.0), (2 / 3)) * pow(HBAR, 2) * pow(density / MASS_P, 5 / 3)) / (5.0 * MASS_E)

    # Pressure Ideal Gas
    P_ig = (K * temp * density) / (mu * MASS_P)

    # Pressure Radiative
    P_rad = (1.0 / 3.0) * RAD_CONST * (temp ** 4.0)

    return P_deg + P_ig + P_rad


# Pressure differentials
@jit(nopython=True)
def dPdp(density, temp):
    # See note above
    # Degenerate
    dPdp_deg = (pow((3.0 * pi ** 2.0), (2 / 3)) * pow(HBAR, 2) * ((density / MASS_P) ** (2.0 / 3.0))) / (
            3.0 * MASS_P * MASS_E)

    # Ideal Gas
    dPdp_ig = (K * temp) / (mu * MASS_P)

    return dPdp_deg + dPdp_ig


@jit(nopython=True)
def dPdT(density, temp):
    # See note above
    dPdT_ig = (density * K) / (mu * MASS_P)
    dPdT_rad = (4.0 / 3.0) * RAD_CONST * (temp ** 3.0)
    return dPdT_ig + dPdT_rad


# Energy generation
@jit(nopython=True)
def epsilon(density, temp):
    # Value for epsilon, used often below

    epp = (1.07e-7) * (density / 1.0e5) * (X ** 2.0) * ((temp / 1.0e6) ** 4.0)
    ecno = (8.24e-26) * (density / 1.0e5) * 0.03 * (X ** 2.0) * ((temp / 1.0e6) ** 19.9)
    return epp + ecno


# Opacity
@jit(nopython=True)
def Kappa(density, temp):
    Kes = 0.02 * (1.0 + X)
    Kff = 1.0e24 * (Z + 0.0001) * ((density / 1.0e3) ** 0.7) * temp ** (-3.5)
    Khminus = 2.5e-32 * (Z / 0.02) * ((density / 1.0e3) ** 0.5) * (temp ** 9.0)
    return ((1.0 / Khminus) + (1.0 / max(Kes, Kff))) ** (-1.0)


# Stellar Structure ODEs
@jit(nopython=True)
def dpdr(radius, mass, density, temp, lum):
    # First last equation in the set of 5 equations in the project description file
    return -((G * mass * density / (radius ** 2.0)) +
             dPdT(density, temp) * dTdr(radius, mass, density, temp, lum)) / (dPdp(density, temp))


@jit(nopython=True)
def dTdr(radius, mass, density, temp, lum):
    # second equation in the set of 5 equations in the project description file
    dTdr_rad = (3.0 * Kappa(density, temp) * density * lum) / (
            16.0 * pi * RAD_CONST * C * (temp ** 3.0) * (radius ** 2.0))
    dTdr_conv = (1.0 - (1.0 / GAMMA)) * (temp / P(density, temp)) * (
            (G * mass * density) / (radius ** 2.0))
    return - min(dTdr_rad, dTdr_conv)


@jit(nopython=True)
def dMdr(radius, density):
    # middle last equation in the set of 5 equations in the project description file
    return 4.0 * pi * (radius ** 2.0) * density


@jit(nopython=True)
def dLdr(radius, density, temp):
    # Second last equation in the set of 5 equations in the project description file
    return dMdr(radius, density) * epsilon(density, temp)


@jit(nopython=True)
def dtaudr(density, temp):
    # Last equation in the set of 5 equations in the project description file
    return Kappa(density, temp) * density


@jit(nopython=True)
def dPdr(radius, mass, density):
    return -(G * mass * density / (radius ** 2.0))


# delta(tau) for optical depth limit
@jit(nopython=True)
def dtau(radius, mass, density, temp, lum):
    return (Kappa(density, temp) * (density ** 2.0)) / (abs(dpdr(radius, mass, density, temp, lum)))


#### NOTE ###
# We need to add the luminosity equation in our section (5.6, eqn 19), and add it as a function here use it

@jit(nopython=True)
def func(dep_var, radius):
    """
    This is the function that bridges the math above to the class and stuff below.
    Given dep_var, a list of depenadant variables, it runs the below functions that
    compile to find T, M, L, tau, and rho.
    :param dep_var: list of floats - dependant variables for current sequence
    :param radius: float - current radius
    :return:
    """

    # Just extracts the array in dep_var to individual variable names
    density, temp, mass, lum, tau = dep_var

    # These functions below call the math functions above and star the chain of math calcualtions.
    # If we assume the class stuff below is correct (and the runge kutta method works), then that
    # leaves the math above to be wrong, which is called and accessed here. Thus the functions
    # dpdr, dTdr, dMdr, dLdr, dtaudr should be inspected (assuming runge kutta works)

    rho = dpdr(radius, mass, density, temp, lum)
    T = dTdr(radius, mass, density, temp, lum)
    M = dMdr(radius, density)
    L = dLdr(radius, density, temp)
    tau = dtaudr(density, temp)

    # returns a list of those values for the next step in the runge kutta sequence
    return np.array([rho, T, M, L, tau])


"""
Below is the class used to find the temp, rho, lum, radius, etc. for a star. This is done every time the SingleStar
class is called.
"""


class SingleStar:

    def __init__(self, dr, rho_c, temp_c, plotmode):
        self.dr = dr
        self.Rstar = 0.0
        self.r = [r0]
        self.d = [rho_c]
        self.t = [temp_c]

        self.m = [(4.0 / 3.0) * pi * (r0 ** 3.0) * rho_c]

        self.l = [self.m[0] * epsilon(rho_c, temp_c)]
        self.tau = [Kappa(rho_c, temp_c) * rho_c]

        # other variables
        self.k = [Kappa(rho_c, temp_c)]
        self.p = [P(rho_c, temp_c)]
        self.dLdr_list = [0.0]

        self.dlogPdlogT = [(temp_c / P(rho_c, temp_c)) *
                           (dPdr(r0, self.m[-1], rho_c) /
                            dTdr(r0, self.m[-1], rho_c, temp_c, self.l[-1]))]

        self.drray = [dr]
        self.dtauarray = []

        self.CreateStar()
        self.RadStar()
        self.Plots(plotmode)

    ###########______Define all Equations______###########

    def rk4(self, y, r, h):
        '''
        This is the runge kutta method we discussed. It runs the function func
        (defined above) on the current dependant variable values and the radius
        y: current values for dependent variables
        r: radius, the independent variables
        h: step-size
        f:function array to be integrated
        '''
        k1 = h * func(y, r)
        k2 = h * func(y + k1 / 2, r + h / 2)
        k3 = h * func(y + k2 / 2, r + h / 2)
        k4 = h * func(y + k3, r + h)

        return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, r + h

    def OpticalDepthLimit(self, new_rk4, r_new):
        """
        This just tests the optical depth to make sure that d_tau is not very small,
        or that the mass hasn't blown up (1e3 Mass sun).
        While those conditions have not been met, returns False
        :param new_rk4: list of floats - rho, temp, mass, luminosity, tau of current radius
        :param r_new: float - new radius
        :return: bool
        """

        # extracts the list into it's values
        cur_d, cur_t, cur_m, cur_l, cur_tau = new_rk4
        d_tau = dtau(r_new, cur_m, cur_d, cur_t, cur_l)

        self.dtauarray.append(d_tau)  # unimportant list

        if d_tau < 0.001 or self.m[-1] > 1e3 * M_SUN:
            return True
        else:
            return False

    def RadStar(self):
        self.Rstar = np.argmin(abs(self.tau[-1] - np.array(self.tau) - (2.0 / 3.0)))
        return self.Rstar

    def do_stuff(self, new_rk4, r_new):
        """

        :param new_rk4: list of floats - eho, temp, mass, luminosity, tau of current radius
        :param r_new: float - new radius
        :return:
        """

        # new_rk4 is a small array that has the distance, temp, mass, lum, and tau. So here we just append those to
        # the corresponding lists
        cur_d, cur_t, cur_m, cur_l, cur_tau = new_rk4
        self.d.append(cur_d)
        self.t.append(cur_t)
        self.m.append(cur_m)
        self.l.append(cur_l)
        self.tau.append(cur_tau)
        self.r.append(r_new)

        # These next lines append to the dLdr list, kappa list, pressure list, and dlogPdlogT lists.
        self.dLdr_list.append(dLdr(r_new, cur_d, cur_t))
        self.k.append(Kappa(cur_d, cur_t))
        self.p.append(P(cur_d, cur_t))
        self.dlogPdlogT.append((cur_t / self.p[-1]) * (
                dPdr(r_new, cur_m, cur_d) / dTdr(r_new, cur_m, cur_d, cur_t, cur_l)))

    def CreateStar(self):

        # these three lines run the runge kutta method (self.rk4) once before entering a loop to keep running it.
        rk4_var = np.array([self.d[-1], self.t[-1], self.m[-1], self.l[-1], self.tau[-1]], float)
        new_rk4, r_new = self.rk4(rk4_var, self.r[-1], self.dr)
        self.do_stuff(new_rk4, r_new)

        # Loop that keeps running the runge kutta method until the Optical limit returns True, then stops.
        while not self.OpticalDepthLimit(new_rk4, r_new):
            new_rk4, r_new = self.rk4(new_rk4, self.r[-1], self.dr)
            self.do_stuff(new_rk4, r_new)

            # Finds the new dr (increment in radius) by checking if the temp is < 5e4
            if self.t[-1] < 5e4:
                self.dr = (0.00001 * r_new) + 1000
            else:
                self.dr = (0.001 * r_new) + 1000

    def Plots(self, plotmode):

        # convert to arrays for normalizations
        r = np.array(self.r)
        rho = np.array(self.d)
        temp = np.array(self.t)
        mass = np.array(self.m)
        lum = np.array(self.l)
        tau = np.array(self.tau)
        pressure = np.array(self.p)
        kappa = np.array(self.k)
        dLdr = np.array(self.dLdr_list)
        star_radius = self.Rstar

        if plotmode:
            # # plot the data
            # plt.figure(1)
            # plt.grid()
            x_axis = r / self.r[-1]
            #
            # plt.plot(x_axis, rho / self.d[0], label='rho')
            # plt.plot(x_axis, temp / self.t[0], label='temp')
            # plt.plot(x_axis, mass / self.m[-1], label='Mass')
            # plt.plot(x_axis, lum / self.l[-1], label='Lum')
            # plt.legend(loc='best', bbox_to_anchor=(0.8, 0.66), prop={'size': 11})
            # # plt.title("Rho", fontsize=25)
            # plt.xlabel("r/R.")
            # plt.ylabel("$\\rho/\\rho_{c}$, $T/T_{c}$, $M/M.$, $L/L.$")
            # plt.savefig(f'Multi-Lined Plot.png', dpi=1000)
            # plt.show()
            # plt.clf()
            # self.plotdata = (x_axis, rho / self.d[0], temp / self.t[0], mass / self.m[-1], lum / self.l[-1])
            # print(self.d[0], "is the rho")
            # print(self.t[0], "is the temp")
            # print(self.t[-1], "is the surface temp")
            # print(self.m[-1], "is the mass")
            # print(self.l[-1], "is the lum")
            # print(self.r[-1], "is the radius")


STEF_BOLT = 5.670373e-8  # Stefan-Boltzmann constant
M_SUN = 1.989e30  # Kg
R_SUN = 6.955e8  # meters
L_SUN = 3.827e26  # watts
pi = np.pi

start = time.time()

class FixDensity:

    def __init__(self, h, temp_c, number_of_stars):
        self.h = h
        self.central_temp = temp_c
        self.number_of_stars = number_of_stars

        self.starA = SingleStar(self.h, 300, temp_c, False)
        self.starB = SingleStar(self.h, 5e5, temp_c, False)
        self.starC = SingleStar(self.h, (300 + 5e5) / 2.0, temp_c, False)

        self.BestStar = self.bisection(self.starA, self.starB, self.starC, 0.02)







    def fix_boundry_condition(self, trialstar):

        RstarIndex = trialstar.Rstar
        Lum1 = trialstar.l[RstarIndex] + (self.number_of_stars * L_SUN)
        # print(Lum1)
        Lum2 = (4.0 * pi * STEF_BOLT * (trialstar.r[RstarIndex] ** 2.0) * (trialstar.t[RstarIndex] ** 4.0))

        return (Lum1 - Lum2) / np.sqrt(Lum1 * Lum2)

    def bisection(self, starA, starB, starC, tol):

        rhoA, rhoB = starA.d[0], starB.d[0]
        while (starB.d[0] - starA.d[0]) > tol:

            if self.fix_boundry_condition(starA) * self.fix_boundry_condition(starC) < 0:
                starB = starC

            else:
                starA = starC

            starCrho = (starA.d[0] + starB.d[0]) / 2.0

            starC = SingleStar(self.h, starCrho, self.central_temp, False)

        starCrho = max(starA.d[0], starB.d[0])
        star_C = SingleStar(self.h, starCrho, self.central_temp, False)
        return star_C




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MAIN SEQUENCE FILE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class MainSequence:
    def __init__(self, NumStars, minTc, maxTc, number_of_stars=0):
        self.NumStars = NumStars
        self.minTc = minTc
        self.maxTc = maxTc
        self.i = 0
        self.number_of_stars = number_of_stars

        self.CreateMS()

    def onestar(self, temp_c):
        start = time.time()
        star = FixDensity(1000.0, temp_c, self.number_of_stars).BestStar
        print(f"Star took {time.time() - start} seconds")
        return star

    @staticmethod
    def STemp(x):
        return ((x.l[x.Rstar]) / (4.0 * np.pi * STEF_BOLT * x.r[x.Rstar] ** 2.0)) ** (1.0 / 4.0)

    def CreateMS(self):
        tempCs = np.linspace(self.minTc, self.maxTc, self.NumStars)
        surfaceTemps = []
        starLums = []
        starmass = []
        star_radius = []
        self.graph_data_without = []
        # If statement for if you have Multiprocessing or not
        if MP_BOOL:
            print("Continuing with Multiprocessing")
            p = multiproc.Pool(multiproc.cpu_count())
            print("Multiprocessing Pool created! Running with all your cores")
            print(GAMMA)
            MS = p.map(self.onestar, tempCs)

            print("Multiprocessing finished, graphing!")
            for i in range(len(MS)):
                # self.graph_data_without.append(MS[i].plotdata)
                surfaceTemps.append(self.STemp(MS[i]))
                starmass.append(MS[i].m[-1])
                star_radius.append(MS[i].r[-1])
                starLums.append(MS[i].l[MS[i].Rstar])
                # plt.scatter(np.log10(surfaceTemps[i]), np.log10(starLums[i] / L_SUN))

        else:
            print("Continuing without Multiprocessing. Running on a single core")
            for i in range(len(tempCs)):
                start = time.time()

                MS = self.onestar(tempCs[i])
                surfaceTemps.append(self.STemp(MS))
                starmass.append(MS.m[-1])
                starLums.append((MS.l[MS.Rstar]))
                print(f"Star {i + 1} took {time.time() - start} seconds")
        #
        # theoretical_mass = []
        # theoretical_radius = []
        # for i in range(len(starmass)):
        #     mass_1 = np.log10(0.35) + (2.62 * np.log10(starmass[i] / M_SUN))
        #     mass_2 = np.log10(1.02) + (3.92 * np.log10(starmass[i] / M_SUN))
        #     theoretical_mass.append(max(mass_1, mass_2))
        #
        #     theoretical_radius_min = (np.log10(1.06) + (.945 * np.log10(starmass[i] / M_SUN)))
        #     theoretical_radius_max = (np.log10(1.33) + (.555 * np.log10(starmass[i] / M_SUN)))
        #     theoretical_radius.append(min(theoretical_radius_min, theoretical_radius_max))


        # plt.scatter(np.log10(np.array(starmass) / M_SUN), np.log10(np.array(starLums) / L_SUN))
        # plt.plot(np.log10(np.array(starmass) / M_SUN), theoretical_mass, "r")
        # plt.title(f"Mass vs Lum for {str(self.number_of_stars)}")
        # plt.xlabel('log Mass (kg)')
        # plt.ylabel('log Luminosity (Watt)')
        # plt.savefig('MS.jpg', dpi=1000)
        # plt.show()
        #
        # plt.scatter(np.log10(np.array(starmass) / M_SUN), np.log10(np.array(star_radius) / R_SUN))
        # plt.plot(np.log10(np.array(starmass) / M_SUN), theoretical_radius, "g")
        # plt.title(f"Mass vs Radius for {str(self.number_of_stars)}")
        # plt.xlabel('log Mass (kg)')
        # plt.ylabel('log Radius (m)')
        # plt.savefig('MS.jpg', dpi=1000)
        # plt.show()

        # self.newtemp = np.log10(surfaceTemps)
        # self.newlum = np.log10(np.array(starLums) / L_SUN)

        # plt.figure(1)

        self.serf_temps = np.log10(surfaceTemps)
        self.star_lums = np.log10(np.array(starLums) / L_SUN)
        self.star_mass = np.log10(np.array(starmass) / M_SUN)
        self.star_radius = np.log10(np.array(star_radius) / R_SUN)
        # plt.figure(2)
        #
        #
        # plt.figure(3)


gammas = [1.9, 1.67, 1.3, 1.2, 1.1]


surf_temp_list = []
star_lums_list = []
star_mass_list = []
star_radius_list = []
for i in gammas:
    GAMMA = i
    a = MainSequence(24, 10 ** 6.6, 10 ** 7.4, 0)

    surf_temp_list.append(a.serf_temps)
    star_lums_list.append(a.star_lums)
    star_mass_list.append(a.star_mass)
    star_radius_list.append(a.star_radius)



plt.scatter(surf_temp_list[2], star_lums_list[2], label=f"gamma = {gammas[2]}")
plt.scatter(surf_temp_list[3], star_lums_list[3], label=f"gamma = {gammas[3]}")
plt.scatter(surf_temp_list[4], star_lums_list[4], label=f"gamma = {gammas[4]}")
plt.title(f"Temp vs Lum for 24 stars with varying gamma factors")
plt.xlabel('logged Temp (K)')
plt.ylabel('logged Lum (Watt)')
plt.legend(loc=1)
plt.grid()
plt.savefig(f'Paige_gamma_plot.jpg', dpi=1000)
plt.gca().invert_xaxis()
plt.show()


plt.figure(1)
plt.scatter(surf_temp_list[0], star_lums_list[0], label=f"gamma = {gammas[0]}")
plt.scatter(surf_temp_list[1], star_lums_list[1], label=f"gamma = {gammas[1]}")
plt.scatter(surf_temp_list[2], star_lums_list[2], label=f"gamma = {gammas[2]}")
plt.scatter(surf_temp_list[3], star_lums_list[3], label=f"gamma = {gammas[3]}")
plt.scatter(surf_temp_list[4], star_lums_list[4], label=f"gamma = {gammas[4]}")
plt.title(f"Temp vs Lum for 24 stars with varying gamma factors")
plt.xlabel('logged Temp (K)')
plt.ylabel('logged Lum (Watt)')
plt.legend(loc=1)
plt.grid()
plt.savefig(f'Paige_gamma_plot.jpg', dpi=1000)
plt.gca().invert_xaxis()
plt.show()

plt.figure(2)
plt.scatter(star_mass_list[0], star_lums_list[0], label=f"gamma = {gammas[0]}")
plt.scatter(star_mass_list[1], star_lums_list[1], label=f"gamma = {gammas[1]}")
plt.scatter(star_mass_list[2], star_lums_list[2], label=f"gamma = {gammas[2]}")
plt.scatter(star_mass_list[3], star_lums_list[3], label=f"gamma = {gammas[3]}")
plt.scatter(star_mass_list[4], star_lums_list[4], label=f"gamma = {gammas[4]}")
plt.title(f"Mass vs Luminosity for 24 stars")
plt.xlabel('logged Mass (kg)')
plt.ylabel('logged Luminosity (Watt)')
plt.grid()
plt.legend(loc=2)
plt.savefig('Paige_gamma_plot_Fig2.jpg', dpi=1000)
plt.show()



plt.figure(3)
plt.scatter(star_mass_list[2], star_lums_list[2], label=f"gamma = {gammas[2]}")
plt.scatter(star_mass_list[3], star_lums_list[3], label=f"gamma = {gammas[3]}")
plt.scatter(star_mass_list[4], star_lums_list[4], label=f"gamma = {gammas[4]}")
plt.title(f"Mass vs Luminosity for 24 stars")
plt.xlabel('logged Mass (kg)')
plt.ylabel('logged Luminosity (Watt)')
plt.grid()
plt.legend(loc=2)
plt.savefig('Paige_gamma_plot_Fig2.jpg', dpi=1000)
plt.show()


plt.figure(4)
plt.scatter(star_mass_list[0], star_radius_list[0], label=f"gamma = {gammas[0]}")
plt.scatter(star_mass_list[1], star_radius_list[1], label=f"gamma = {gammas[1]}")
plt.scatter(star_mass_list[2], star_radius_list[2], label=f"gamma = {gammas[2]}")
plt.scatter(star_mass_list[3], star_radius_list[3], label=f"gamma = {gammas[3]}")
plt.scatter(star_mass_list[4], star_radius_list[4], label=f"gamma = {gammas[4]}")
plt.title(f"Mass vs Radius for 24 stars")
plt.xlabel('logged Mass (kg)')
plt.ylabel('logged Radius (m)')
plt.grid()
plt.legend(loc=2)
plt.savefig('Paige_gamma_plot_Fig3.jpg', dpi=1000)
plt.show()





plt.figure(5)
plt.scatter(star_mass_list[2], star_radius_list[2], label=f"gamma = {gammas[2]}")
plt.scatter(star_mass_list[3], star_radius_list[3], label=f"gamma = {gammas[3]}")
plt.scatter(star_mass_list[4], star_radius_list[4], label=f"gamma = {gammas[4]}")
plt.title(f"Mass vs Radius for 24 stars")
plt.xlabel('logged Mass (kg)')
plt.ylabel('logged Radius (m)')
plt.grid()
plt.legend(loc=2)
plt.savefig('Paige_gamma_plot_Fig3.jpg', dpi=1000)
plt.show()

# b = MainSequence(8, 10 ** 6.6, 10 ** 7.4, 0.1)
# # c = MainSequence(16, 10 ** 6.6, 10 ** 7.4, 0.01)
# # d = MainSequence(16, 10 ** 6.6, 10 ** 7.4, 0.001)
# # e = MainSequence(16, 10 ** 6.6, 10 ** 7.4, 0.0001)
# for p in range(len(a.graph_data_without)):
#     x_axis, rho, temp, mass, lum = a.graph_data_without[p]
#     # x_axis_lum, rho_lum, temp_lum, mass_lum, lum_lum = b.graph_data_without[p]
#     # plt.scatter(a.newtemp, a.newlum, label=f"{a.number_of_stars}$_\odot$")
#     # plt.scatter(b.newtemp, b.newlum, label=f"{b.number_of_stars}$_\odot$")
#     # plt.scatter(c.newtemp, c.newlum, label=f"{c.number_of_stars}$_\odot$")
#     # plt.scatter(d.newtemp, d.newlum, label=f"{d.number_of_stars}$_\odot$")
#     # plt.scatter(e.newtemp, e.newlum, label=f"{e.number_of_stars}$_\odot$")
#
#
#     plt.plot(x_axis, rho, 'r', label='rho')
#     plt.plot(x_axis, temp, 'g',  label='temp')
#     plt.plot(x_axis, mass, 'b', label='Mass')
#     plt.plot(x_axis, lum, 'm', label='Lum')
#
#     # plt.plot(x_axis_lum, rho_lum, 'k--')
#     # plt.plot(x_axis_lum, temp_lum, 'k--')
#     # plt.plot(x_axis_lum, mass_lum, 'k--')
#     # plt.plot(x_axis_lum, lum_lum, 'k--')
#     plt.legend(loc='best', bbox_to_anchor=(0.8, 0.66), prop={'size': 11})
#     # plt.title("Rho", fontsize=25)
#     plt.title("Normalized properties of stellar structure with and without stellar warming")
#     plt.xlabel("r/R$_\odot$")
#     plt.ylabel("$\\rho/\\rho_{c}$, $T/T_{c}$, $M/M_\odot$, $L/L_\odot$")
#     plt.savefig(f'Multi-Lined Plot.png', dpi=1000)
#     plt.show()



# plt.gca().invert_xaxis()
# plt.title(f"Temp vs Lum for Stellar warming between 0.0001$L_\odot$ to 1$_\odot$")
# plt.xlabel('log Temperature (K)')
# plt.ylabel('log Luminosity (Watt)')
# plt.legend(loc='best', bbox_to_anchor=(0.8, 0.66), prop={'size': 11})
# plt.show()



print("It took:", (time.time() - start) // 60, " minutes and", (time.time() - start) % 60, "seconds to run!")