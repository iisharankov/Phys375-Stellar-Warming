from numba import jit, jitclass
import numba
import const as ct
import numpy as np
import matplotlib.pyplot as plt
import time

mu = (2.0 * ct.X + 0.75 * ct.Y + 0.5 * ct.Z) ** (-1.0)
r0 = 0.001  # m
S = 1.0  # error tolerance


# Pressures
@jit(nopython=True)
def P(density, temp):
    P_deg = (((3.0 * np.pi ** 2.0) ** (2.0 / 3.0)) * (ct.HBAR ** (2.0)) * ((density / ct.MASS_P) ** (5.0 / 3.0))) / (
            5.0 * ct.MASS_E)

    P_ig = (density * ct.K * temp) / (mu * ct.MASS_P)

    P_rad = (1.0 / 3.0) * ct.RAD_CONST * (temp ** 4.0)

    return P_deg + P_ig + P_rad


# Pressure differentials
@jit(nopython=True)
def dPdp(density, temp):
    dPdp_deg = (((3.0 * np.pi ** 2.0) ** (2.0 / 3.0)) * (ct.HBAR ** 2.0) * ((density / ct.MASS_P) ** (2.0 / 3.0))) / (
            3.0 * ct.MASS_P * ct.MASS_E)

    dPdp_ig = (ct.K * temp) / (mu * ct.MASS_P)

    return dPdp_deg + dPdp_ig

@jit(nopython=True)
def dPdT(density, temp):
    dPdT_ig = (density * ct.K) / (mu * ct.MASS_P)
    dPdT_rad = (4.0 / 3.0) * ct.RAD_CONST * (temp ** 3.0)

    return dPdT_ig + dPdT_rad


# Energy generation
@jit(nopython=True)
def epsilon(density, temp):
    epp = (1.07e-7) * (density / 1.0e5) * (ct.X ** 2.0) * ((temp / 1.0e6) ** 4.0)

    ecno = (8.24e-26) * (density / 1.0e5) * 0.03 * (ct.X ** 2.0) * ((temp / 1.0e6) ** 19.9)

    return epp + ecno


# Opacity
@jit(nopython=True)
def Kappa(density, temp):
    Kes = 0.02 * (1.0 + ct.X)

    Kff = 1.0e24 * (ct.Z + 0.0001) * ((density / 1.0e3) ** 0.7) * temp ** (-3.5)

    Khminus = 2.5e-32 * (ct.Z / 0.02) * ((density / 1.0e3) ** 0.5) * (temp ** 9.0)

    return ((1.0 / Khminus) + (1.0 / max(Kes, Kff))) ** (-1.0)


@jit(nopython=True)
def dTdr(radius, mass, density, temp, lum):

    dTdr_rad = (3.0 * Kappa(density, temp) * density * lum) / (
                16.0 * np.pi * ct.RAD_CONST * ct.C * (temp ** 3.0) * (radius ** 2.0))

    dTdr_conv = (1.0 - (1.0 / ct.GAMMA)) * (temp / P(density, temp)) * (
                (ct.G * mass * density) / (radius ** 2.0))

    return - min(dTdr_rad, dTdr_conv)


# Stellar Structure ODEs
@jit(nopython=True)
def dpdr(radius, mass, density, temp, lum):

    return -((ct.G * mass * density / (radius ** 2.0)) +
             dPdT(density, temp) * dTdr(radius, mass, density, temp, lum)) / (dPdp(density, temp))


@jit(nopython=True)
def dMdr(radius, density):
    return 4.0 * np.pi * (radius ** 2.0) * density


@jit(nopython=True)
def dLdr(radius, density, temp):
    return dMdr(radius, density) * epsilon(density, temp)

@jit(nopython=True)
def dtaudr(density, temp):
    return Kappa(density, temp) * density

@jit(nopython=True)
def dPdr(radius, mass, density):
    return -(ct.G * mass * density / (radius ** 2.0))


# delta(tau) for optical depth limit
@jit(nopython=True)
def dtau(radius, mass, density, temp, lum):
    return (Kappa(density, temp) * (density ** 2.0)) / (abs(dpdr(radius, mass, density, temp, lum)))



@jit(nopython=True)
def func(dep_var, r):

    radius = r
    density, temp, mass, lum, tau = dep_var

    rho = dpdr(radius, mass, density, temp, lum)
    T = dTdr(radius, mass, density, temp, lum)
    M = dMdr(radius, density)
    L = dLdr(radius, density, temp)
    tau = dtaudr(density, temp)

    return np.array([rho, T, M, L, tau])


# @jit(nopython=True)
# def rk4_external(y, r, h):
#     '''
#     y: current values for dependent variables
#     r: radius, the independent variables
#     h: step-size
#     f:function array to be integrated
#     '''
#
#     k1 = h * f(y, r)
#     k2 = h * f(y + k1/2, r + h/2)
#     k3 = h * f(y + k2/2, r + h/2)
#     k4 = h * f(y + k3, r + h)
#
#     return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, r + h


class SingleStar:

    def __init__(self, dr, rho_c, temp_c, plotmode):
        self.plotmode = plotmode
        self.dr = dr
        self.Rstar = 0.0

        # lists to hold the values from rk4
        # stellar structurs variables
        self.r = [r0]
        self.d = [rho_c]
        self.t = [temp_c]

        self.m = [(4.0 / 3.0) * np.pi * (r0 ** 3.0) * rho_c]

        self.l = [self.m[0] * epsilon(rho_c, temp_c)]
        self.tau = [Kappa(rho_c, temp_c) * rho_c]

        # other variables
        self.k = [Kappa(rho_c, temp_c)]
        self.p = [P(rho_c, temp_c)]
        self.dLdr_list = [0.0]

        self.dlogPdlogT = \
            [(temp_c / P(rho_c, temp_c)) * (dPdr(r0, self.m[-1], rho_c) /
                                            dTdr(r0, self.m[-1], rho_c, temp_c, self.l[-1]))]

        self.drray = [dr]
        self.dtaurray = []

        self.CreateStar()
        self.RadStar()
        self.Plots(plotmode)

    ###########______Define all Equations______###########


    def rk4(self, y, r, h, f):
        '''
        y: current values for dependent variables
        r: radius, the independent variables
        h: step-size
        f:function array to be integrated
        '''
        k1 = h * f(y, r)
        k2 = h * f(y + k1/2, r + h/2)
        k3 = h * f(y + k2/2, r + h/2)
        k4 = h * f(y + k3, r + h)

        return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, r + h

    # The following three functions as for the assignment 4 question to test whether rk4 is doing what it's supposed to!
    # def ftest(self, dep_var, r):
    #     density = dep_var[0]
    #     mass = dep_var[1]
    #
    #     rho = -density * mass / pow(r, 2)
    #     M = 4 * np.pi * pow(r, 2) * density
    #
    #     return np.array([rho, M], float)
    def OpticalDepthLimit(self):

        d_tau = dtau(self.r[-1], self.m[-1], self.d[-1], self.t[-1], self.l[-1])
        self.dtaurray.append(d_tau)
        if d_tau < 0.0001 or self.m[-1] > (1.0e3) * ct.M_SUN:
            return True
        else:
            return False

    def RadStar(self):

        self.Rstar = np.argmin(abs(self.tau[-1] - np.array(self.tau) - (2.0 / 3.0)))

        if self.tau[self.Rstar] == 0:
            self.Rstar = len(self.tau) - 1

        #        print "Radius of the star is:", self.r[self.Rstar]
        return self.Rstar

    def do_stuff(self, new_rk4, new_radius):
        self.d.append(new_rk4[0])
        self.t.append(new_rk4[1])
        self.m.append(new_rk4[2])
        self.l.append(new_rk4[3])
        self.tau.append(new_rk4[4])
        self.r.append(new_radius)
        self.dLdr_list.append(dLdr(self.r[-1], self.d[-1], self.t[-1]))
        self.k.append(Kappa(self.d[-1], self.t[-1]))
        self.p.append(P(self.d[-1], self.t[-1]))
        self.dlogPdlogT.append((self.t[-1] / self.p[-1]) * (
                dPdr(self.r[-1], self.m[-1], self.d[-1]) / dTdr(self.r[-1], self.m[-1], self.d[-1],
                                                                          self.t[-1], self.l[-1])))

    def CreateStar(self):

        rk4_var = np.array([self.d[-1], self.t[-1], self.m[-1], self.l[-1], self.tau[-1]], float)
        new_rk4, new_radius = self.rk4(rk4_var, self.r[-1], self.dr, func)
        self.do_stuff(new_rk4, new_radius)

        while not self.OpticalDepthLimit():
            new_rk4, new_radius = self.rk4(new_rk4, self.r[-1], self.dr, func)
            self.do_stuff(new_rk4, new_radius)

            if self.t[-1] < 50.0e3:
                self.dr = (0.00001 * new_radius) + 1000
            else:
                self.dr = (0.001 * new_radius) + 1000

    # def CreateStar(self):
    #
    #     rk4_var = np.array([self.d[-1], self.t[-1], self.m[-1], self.l[-1], self.tau[-1]], float)
    #     new_rk4, new_radius = self.rk4(rk4_var, self.r[-1], self.dr, func)
    #     self.do_stuff(new_rk4, new_radius)
    #
    #     while not self.OpticalDepthLimit():
    #         new_rk4, new_radius = self.rk4(new_rk4, self.r[-1], self.dr, func)
    #         self.do_stuff(new_rk4, new_radius)
    #
    #         if self.t[-1] < 50.0e3:
    #             self.dr = (0.00001 * new_radius) + 1000
    #         else:
    #             self.dr = (0.001 * new_radius) + 1000

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

        if plotmode:
            # plot the data
            plt.figure(1)
            plt.grid()
            x_axis = r / self.r[-1]
            plt.plot(x_axis, rho / self.d[0], label='rho')
            plt.plot(x_axis, temp / self.t[0], label='temp')
            plt.plot(x_axis, mass / self.m[-1], label='Mass')
            plt.plot(x_axis, lum / self.l[-1], label='Lum')
            plt.legend(loc='best', bbox_to_anchor=(0.8, 0.66), prop={'size': 11})
            plt.title("Rho", fontsize=25)
            plt.savefig('Fig1.png')
            plt.show()
            # '''
            # plt.figure(2)
            # plt.grid()
            # plt.plot(x_axis, temp/self.t[0], label='temp')
            # plt.title("Temp", fontsize=25)
            # plt.show()
            #
            # plt.figure(3)
            # plt.grid()
            # plt.plot(x_axis, mass/self.m[-1], label='Mass')
            # plt.title("Mass", fontsize=25)
            # plt.show()
            #
            # plt.figure(4)
            # plt.grid()
            # plt.plot(x_axis, lum/self.l[-1], label='Lum')
            # plt.title("Lum", fontsize=25)
            # plt.show()
            # '''
            #
            # plt.figure(9)
            # plt.grid()
            # plt.plot(r / self.r[self.Rstar], tau / self.tau[-1], label='Tau')
            # plt.title("Tau", fontsize=25)
            # plt.savefig('Tau.png', dpi=1000)
            # plt.show()
            #
            # plt.figure(5)
            # plt.grid()
            # plt.plot(r / self.r[self.Rstar], dLdr / (self.l[-1] / self.r[self.Rstar]), label='dL/dR')
            # plt.title("dLdR", fontsize=25)
            # plt.savefig('dLdR.png', dpi=1000)
            # plt.show()
            #
            # plt.figure(6)
            # plt.grid()
            # plt.legend(loc='best', bbox_to_anchor=(0.8, 0.66), prop={'size': 11})
            # plt.plot(r / self.r[self.Rstar], pressure / self.p[0], label='Pressure')
            # plt.title("Pressure", fontsize=25)
            # plt.savefig('Pressure.png', dpi=1000)
            # plt.show()
            #
            # plt.figure(7)
            # plt.grid()
            # plt.plot(r / self.r[self.Rstar], np.log10(kappa), label='Opacity')
            # plt.title("Opacity", fontsize=25)
            # plt.savefig('Opacity.png', dpi=1000)
            # plt.show()
            #
            # plt.figure(8)
            # axes = plt.gca()
            # # axes.set_xlim(0,11)
            # axes.set_ylim(0, 10)
            # plt.grid()
            # plt.plot(r / self.r[self.Rstar], self.dlogPdlogT, label='dlogP/dlogT')
            # plt.title("dlogP/dlogT", fontsize=25)
            # plt.savefig('dlogP-dlogT.png', dpi=1000)
            # plt.show()


start = time.time()
SingleStar(1000.0,0.5e3,1.5e7, 1)
print ("It took:", (time.time()-start)//60, " minutes and", (time.time()-start)%60 ,"seconds to run!")

