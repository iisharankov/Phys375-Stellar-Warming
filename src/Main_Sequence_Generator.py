import numpy as np
import time
import matplotlib.pyplot as plt
import const as ct
from src.Star_Creator import SingleStar

# Tries to import multiprocessing, if failed, MP_BOOL is False for later use
try:
    import multiprocessing as multiproc
    MP_BOOL = True
except:
    MP_BOOL = False




STEF_BOLT = 5.670373e-8  # Stefan-Boltzmann constant
L_SUN = 3.827e26  # watts


start = time.time()

class FixDensity:

    def __init__(self, h, temp_c):
        self.h = h
        self.central_temp = temp_c

        self.starA = SingleStar(self.h, 0.3e3, temp_c, False)
        self.starB = SingleStar(self.h, 500.0e3, temp_c, False)
        self.starC = SingleStar(self.h, (0.3e3 + 500.0e3) / 2.0, temp_c, False)
        # self.BestStar = self.starC
        self.BestStar = self.bisection(self.starA, self.starB, self.starC, 0.01)

    def f(self, trialstar):

        RstarIndex = trialstar.Rstar
        Lum1 = trialstar.l[RstarIndex]
        Lum2 = (4.0 * np.pi * STEF_BOLT * (trialstar.r[RstarIndex] ** 2.0) * (trialstar.t[RstarIndex] ** 4.0))

        return (Lum1 - Lum2) / np.sqrt(Lum1 * Lum2)
    #
    def bisection(self, starA, starB, starC, tol):

        while (starB.d[0] - starA.d[0]) / 2.0 > tol:

            if starC == 0:
                return starC

            elif self.f(starA) * self.f(starC) < 0:
                starB = starC

            else:
                starA = starC

            starCrho = (starA.d[0] + starB.d[0]) / 2.0

            starC = SingleStar(self.h, starCrho, self.central_temp, False)

        starCrho = max(starA.d[0], starB.d[0])
        starC = SingleStar(self.h, starCrho, self.central_temp, False)
        return starC


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MAIN SEQUENCE FILE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class MainSequence:
    def __init__(self, NumStars, minTc, maxTc):
        self.NumStars = NumStars
        self.minTc = minTc
        self.maxTc = maxTc
        self.i = 0

        self.CreateMS()


    def onestar(self, temp_c):
        start = time.time()
        star = FixDensity(1000.0, temp_c).BestStar
        print(f"Star took {time.time() - start} seconds")
        return star

    @staticmethod
    def STemp(x):
        return ((x.l[x.Rstar]) / (4.0 * np.pi * STEF_BOLT * x.r[x.Rstar] ** 2.0)) ** (1.0 / 4.0)

    def CreateMS(self):
        tempCs = np.linspace(self.minTc, self.maxTc, self.NumStars)
        surfaceTemps = []
        starLums = []

        # If statement for if you have Multiprocessing or not
        if  MP_BOOL:
            print("Continuing with Multiprocessing")
            p = multiproc.Pool(multiproc.cpu_count())
            print("Multiprocessing Pool created! Running with all your cores")
            MS = p.map(self.onestar, tempCs)

            print("Multiprocessing finished, graphing!")
            for i in range(len(MS)):
                surfaceTemps.append(self.STemp(MS[i]))
                starLums.append(MS[i].l[MS[i].Rstar])
                plt.scatter(np.log10(surfaceTemps[i]), np.log10(starLums[i] / L_SUN))

        else:
            print("Continuing without Multiprocessing. Running on a single core")

            for i in range(len(tempCs)):
                start = time.time()

                MS = self.onestar(tempCs[i])
                surfaceTemps.append(self.STemp(MS))
                starLums.append(MS.l[MS.Rstar])
                print(f"Star {i + 1} took {time.time() - start} seconds")
                plt.scatter(np.log10(surfaceTemps[i]), np.log10(starLums[i] / L_SUN))

        plt.gca().invert_xaxis()
        plt.xlabel('log Temperature (K)')
        plt.ylabel('log Luminosity (Watt)')
        plt.savefig('MS.jpg', dpi=1000)
        plt.show()


MainSequence(30, 10 ** 6.6, 10 ** 7.4)

print("It took:", (time.time() - start) // 60, " minutes and", (time.time() - start) % 60, "seconds to run!")