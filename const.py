import multiprocessing

# General Constants
MASS_E = 9.10938291e-31  # electron mass
MASS_P = 1.67262178e-27  # proton mass
STEF_BOLT = 5.670373e-8  # Stefan-Boltzmann constant
K_BOLT = 1.381e-23  # Boltzmann constant
C = 299792458  # speed of light
G = 6.673e-11  # Gravitation constant
HBAR = 1.054571817e-34

# Astronomy constants
RAD_CONST= (4*STEF_BOLT)/C  # radiation constant
FRAC_X = 0.73  # Hydrogen mass fraction
FRAC_Y = 0.25  # Helium mass fraction
FRAC_Z = 0.02  # Metals mass fraction)
GAMMA = 5/3  # adiabatic constant
M_SUN = 1.989e30  # Kg
R_SUN = 6.955e8  # meters
L_SUN = 3.827e26  # watts


# Other
NUM_OF_CORES = multiprocessing.cpu_count()