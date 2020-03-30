import numpy as np
from numba import jit, jitclass
import numba
import multiprocessing as multiproc
import time
import matplotlib.pyplot as plt
import const as ct
from src.reg_star_functions import SingleStar

start = time.time()


class FixDensity:

    def __init__(self, h, temp_c):
        self.h = h
        self.central_temp = temp_c

        self.starA = SingleStar(self.h, 0.3e3, temp_c, False)

        self.starB = SingleStar(self.h, 500.0e3, temp_c, False)

        self.starC = SingleStar(self.h, (0.3e3 + 500.0e3) / 2.0, temp_c, False)

        self.BestStar = self.bisection(self.starA, self.starB, self.starC, 0.01)

    def f(self, trialstar):

        RstarIndex = trialstar.Rstar

        Lum1 = trialstar.l[RstarIndex]
        Lum2 = (4.0 * np.pi * ct.STEF_BOLT * (trialstar.r[RstarIndex] ** 2.0) * (trialstar.t[RstarIndex] ** 4.0))

        return (Lum1 - Lum2) / np.sqrt(Lum1 * Lum2)

    def bisection(self, starA, starB, starC, tol):

        starCrho = (starA.d[0] + starB.d[0]) / 2.0


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


# FixDensity(1000.0,1.5e7)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MAIN SEQUENCE FILE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class MainSequence:
    def __init__(self, NumStars, minTc, maxTc):
        self.NumStars = NumStars
        self.minTc = minTc
        self.maxTc = maxTc
        self.i = 0
        self.CreateMS()


    def onestar(self, temp_c):
        self.i += 1
        star = FixDensity(1000.0, temp_c).BestStar
        print(f"Star {self.i}/{self.NumStars}")
        return star

    @staticmethod
    def STemp(x):
        return ((x.l[x.Rstar]) / (4.0 * np.pi * ct.STEF_BOLT * x.r[x.Rstar] ** 2.0)) ** (1.0 / 4.0)



    def mp_worker(self, two_stuff):
        inputs, the_time = two_stuff
        print(f"Processs {inputs}. Waiting {the_time} seconds")
        time.sleep(int(the_time))
        print(f"Process {inputs} DONE")

    # def mp_handler(self):
    #     p = multiproc.Pool(multiproc.cpu_count())
    #     p.map(mp_worker, data)



    def CreateMS(self):
        tempCs = np.linspace(self.minTc, self.maxTc, self.NumStars)
        surfaceTemps = []
        starLums = []

        p = multiproc.Pool(multiproc.cpu_count())
        MS = p.map(self.onestar, tempCs)


        for i in range(len(MS)):
            surfaceTemps.append(self.STemp(MS[i]))
            starLums.append(MS[i].l[MS[i].Rstar])

        # for i in range(len(tempCs)):
        #     start = time.time()
        #     MS = self.onestar(tempCs[i])
        #     print(f"Star: {i+1}")
        #     surfaceTemps.append(self.STemp(MS))
        #     starLums.append(MS.l[MS.Rstar])

            plt.scatter(surfaceTemps[i], starLums[i] / ct.L_SUN)
            # plt.scatter(np.log10(surfaceTemps[i]), np.log10(starLums[i] / ct.L_SUN))

        plt.gca().invert_xaxis()
        plt.xlabel('log Temperature (K)')
        plt.ylabel('log Luminosity (Watt)')
        plt.savefig('MS.jpg', dpi=1000)
        plt.show()


MainSequence(100, 10 ** 6.6, 10 ** 7.4)

print("It took:", (time.time() - start) // 60, " minutes and", (time.time() - start) % 60, "seconds to run!")