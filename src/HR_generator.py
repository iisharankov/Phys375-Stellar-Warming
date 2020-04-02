import numpy as np
import matplotlib.pyplot as plt

######################### Parameters to Vary ###################
rho_center = 150000  ## Parameters for sun chosen
T_center = 1.57e7
X = 0.7381
Z = 0.0134

Y = 1 - X - Z
################################################################


####################### Constants ##################
Rsun = 696340000
Msun = 1.989e30
mp = 1.67e-27
G = 6.67e-11
k = 1.38e-23
hbar = 1.0545718e-34
me = 9.1093837015e-31
u = (2 * X + 0.75 * Y + 0.5 * Z) ** -1  # approximation for ionized gas? u = 2X + 0.75Y + 0.5Z
sigma = 5.67e-8
c = 2.998e8
gamma = 5 / 3  # Not sure what this is
a = 4 * sigma / c
###################################################

###################################################
r = np.linspace(0.001, Rsun, 5000)

rho = np.zeros(5000)
rho[0] = rho_center

T = np.zeros(5000)
T[0] = T_center

L = np.zeros(5000)
L[0] = 0

M = np.zeros(5000)
M[0] = 0

Tau = np.zeros(5000)
Tau[0] = 0

dr = r[1] - r[0]

###################################################

i = 0
while i < 4999 or M[i] > 10 ** 3 * Msun:
    i += 1

    # Pressure

    P = (3 * np.pi ** 2) ** (2 / 3) * hbar ** 2 / (5 * me) * (rho[i - 1] / mp) ** (5 / 3) + rho[i - 1] * k * T[
        i - 1] / (u * mp) + 1 / 3 * a * T[i - 1] ** 4

    ########## Kappa ###########

    Kappa_ff = 1e24 * (Z + 0.0001) * (rho[i - 1] / 10 ** 3) ** 0.7 * T[i - 1] ** (-3.5)
    Kappa_es = 0.02 * (1 + X)
    Kappa_Hminus = 2.5e-32 * (Z / 0.02) * (rho[i - 1] / 10 ** 3) ** 0.5 * T[i - 1] ** 9

    Kappa = (1 / Kappa_Hminus + 1 / max(Kappa_es, Kappa_ff)) ** -1

    case1 = 3 * Kappa * rho[i - 1] * L[i - 1] / (16 * np.pi * a * c * T[i - 1] ** 3 * r[i - 1] ** 2)
    case2 = (1 - 1 / gamma) * T[i - 1] / P * G * M[i - 1] * rho[i - 1] / r[i - 1] ** 2

    ########### rho ################

    dP_rho = (3 * np.pi ** 2) ** (2 / 3) * hbar ** 2 / (3 * me * mp) * (rho[i - 1] / mp) ** (2 / 3) + k * T[i - 1] / (
                u * mp)  # dP/drho

    dP_T = rho[i - 1] * k / (u * mp) + 4 / 3 * a * T[i - 1] ** 3  # dP/dT

    rho[i] = rho[i - 1] - dr * ((G * M[i - 1] * rho[i - 1] / r[i - 1] ** 2 - dP_T * (min(case1, case2))) / dP_rho)

    ########## M ###############

    M[i] = M[i - 1] + (4 * np.pi * rho[i - 1] * r[i - 1] ** 2) * dr

    ########## T ###############

    T[i] = T[i - 1] - dr * (min(case1, case2))

    ########## L ###############

    E_pp = 1.07e-7 * rho[i - 1] / 10 ** 5 * X ** 2 * (T[i - 1] / 10 ** 6) ** 4
    E_cno = 8.24e-26 * rho[i - 1] / 10 ** 5 * 0.03 * X ** 2 * (T[i - 1] / 10 ** 6) ** 19.9
    E = E_pp + E_cno

    L[i] = L[i - 1] + dr * (4 * np.pi * r[i - 1] ** 2 * E)

    ########## Tau #############

    Tau[i] = Tau[i - 1] + dr * (Kappa * rho[i - 1])  ##move

    # cond = Kappa * rho[i] ** 2 / abs(
    #     ((G * M[i - 1] * rho[i - 1] / r[i - 1] ** 2 - dP_T * (min(case1, case2))) / dP_rho))

    # print (Tau[i])
    # print (M[i]/Msun)
    # print(cond)
plt.plot(r,rho,"g-")
plt.show()

plt.plot(r, M, 'g-')
plt.show()
