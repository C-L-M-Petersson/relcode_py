# ==================================================================================================
# This file contains helper functions that are shared among several parts of the
# atomicsystem-scripts.
# Examples are kappa <-> l,j and wigner 3j-symbol functions
# ==================================================================================================
import numpy as np
from sympy import N as sympy_to_num
from sympy.physics.wigner import wigner_3j
from scipy.special import gamma
from scipy.constants import fine_structure
import glob
import json
from scipy.interpolate import InterpolatedUnivariateSpline as interp


# ==================================================================================================
#
# ==================================================================================================
def get_one_photon_directory_metadata(data_dir):
    # This function just parses the Fortran output data directory for
    # the folders called pert_<kappa>_<number>.
    # It then gives back the directory name and kappa as a tuple (kappa, dir_name, n)
    # where n is the principal quantum number calculated from the last number k of the pert dirs,
    # which is k = n-l
    globbed_pert_dirs = glob.glob(data_dir + "pert_*")
    globbed_without_old = []
    # We don't want any "old" pert data directories
    for globbed_dir in globbed_pert_dirs:
        if globbed_dir.find("old") == -1:
            globbed_without_old.append(globbed_dir)

    # print(globbed_pert_dirs)

    tuples = []
    for dir in globbed_without_old:
        pert_only = dir[len(data_dir) :]
        N = len("pert_")
        strip_pert = pert_only[N:]
        end_only = strip_pert[-1:]
        end_int = int(end_only)
        # print(end_only)
        strip_end = strip_pert[:-2]
        kappa_str = strip_end
        kappa = int(kappa_str)
        l = l_from_kappa(kappa)
        n = end_int + l
        kappa_pert_tuple = (pert_only, kappa, n)
        # print(kappa_pert_tuple)
        tuples.append(kappa_pert_tuple)

    print(tuples)

    return tuples


# ==================================================================================================
#
# ==================================================================================================
class IonHole:
    def __init__(self, kappa, n_qn, binding_energy):
        self.kappa = kappa
        self.n = n_qn  # n quantum number (principal)
        self.l = l_from_kappa(kappa)
        self.j = j_from_kappa(kappa)
        self.name = (
            str(self.n) + l_to_str(self.l) + ("_{%i/2}" % (j_from_kappa_int(kappa)))
        )
        self.binding_energy = binding_energy


# ==================================================================================================
#
# ==================================================================================================
def load_raw_data(path):
    return np.loadtxt(path)


def kappa_from_l_and_j(l, j):
    if int(2 * l) == int(2 * j) - 1:
        return -(l + 1)
    else:
        return l


def l_from_kappa(kappa):
    if kappa < 0:
        return -kappa - 1
    else:
        return kappa


def phase(x):
    """Returns 1 if the input is even and -1 if it is odd. Mathematically equivalent to (-1)^x"""
    if x % 2 == 0:
        return 1
    else:
        return -1


def mag(x):
    """Returns the absolute value squared of the input"""
    return np.abs(x) ** 2


def cross(x, y):
    """Returns the 'cross term' between x and y: 2Re(x*y^dagger)"""
    return 2 * np.real(x * np.conjugate(y))


def exported_mathematica_tensor_to_python_list(string):
    return json.loads(string.replace("{", "[").replace("}", "]").replace("\n", ""))


def j_from_kappa(kappa):
    l = l_from_kappa(kappa)
    if kappa < 0:
        return l + 0.5
    else:
        return l - 0.5


def interpolated(x, y, number_of_datapoints):
    """Returns the new x and y values of the 1D-function described by the interpolation
    of the input x and y data points. Interpolates the function values, y, in the domain, x, to
    the given number of datapoints"""
    new_x = np.linspace(x[0], x[-1], number_of_datapoints)
    return new_x, interp(x, y)(new_x)


def j_from_kappa_int(kappa):
    l = l_from_kappa(kappa)
    if kappa < 0:
        return 2 * l + 1
    else:
        return 2 * l - 1


def l_from_str(l_str):
    l = -1

    if l_str == "s":
        l = 0
    elif l_str == "p":
        l = 1
    elif l_str == "d":
        l = 2
    elif l_str == "f":
        l = 3

    if l == -1:
        raise ValueError(
            "l_from_str(): invalid or unimplemented string for l quantum number."
        )
    else:
        return l


def l_to_str(l):
    if l == 0:
        return "s"
    elif l == 1:
        return "p"
    elif l == 2:
        return "d"
    elif l == 3:
        return "f"
    elif l == 4:
        return "g"
    elif l == 5:
        return "h"
    else:
        raise ValueError(
            "l_to_str(): invalid or unimplemented string for l quantum number."
            "Function was given l =",
            l,
        )


# ==================================================================================================
#
# ==================================================================================================
def wigner3j_numerical(hole_kappa, final_kappa, mj):
    mjj = int(2 * mj)
    j_hole = j_from_kappa_int(hole_kappa)
    j_final = j_from_kappa_int(final_kappa)
    if mjj > j_hole or mjj > j_final:
        return
    # print("j_hole, j_final, mj: %i/2 %i/2 %i/2" % (j_hole, j_final, mjj))
    K = 1
    q = 0
    w3j = wigner_3j(j_final / 2, K, j_hole / 2, -mjj / 2, q, mjj / 2)
    # print(w3j, sympy_to_num(w3j))
    return sympy_to_num(w3j)


def wigner3j_numerical2(j_hole, j_final, mjj):
    # print("j_hole, j_final, mj: %i/2 %i/2 %i/2" % (j_hole, j_final, mjj))
    K = 1
    q = 0
    w3j = wigner_3j(j_final / 2, K, j_hole / 2, -mjj / 2, q, mjj / 2)
    # print(w3j, sympy_to_num(w3j))
    return sympy_to_num(w3j)


def wigner_eckart_phase(final_kappa, mj):
    return np.power(-1.0, (j_from_kappa(final_kappa) - mj))


def coulomb_phase(kappa, energy, Z, use_relativistic_wavenumber=True):
    """This is the definition of the phase of the Coulomb function,
    both the angular momentum part and the so-called Coulomb phase.
    Electron energy should be given in atomic units.
    This formula uses the relativistic version of the wavenumber by default.
    If you want to use the nonrelativistic version, pass in
    use_relativistic_wavenumber=False."""

    if kappa == 0:
        # A kappa value of zero is unphysical. However we will call this function with zero kappa
        # values often as part of the analysis, so we just return a phase of zero for that case
        return np.zeros(len(energy))

    l = l_from_kappa(kappa)

    if use_relativistic_wavenumber:
        k = wavenumber(energy)
    else:
        k = np.sqrt(2 * energy)

    x = Z / k
    b = np.angle(gamma(l + 1 - 1j * x))

    return b - l * np.pi / 2


def wavenumber(energy):
    """Returns the relativistic wave number (k-value)."""
    fsc_inv = 1.0 / fine_structure
    return np.sqrt((energy + fsc_inv**2) ** 2 - fsc_inv**4) * fine_structure


# ==================================================================================================
#
# ==================================================================================================
def convert_rate_to_cross_section(rates, omegas, divide=True):
    N = len(omegas)
    cm2 = (0.52917721092**2) * 100.0
    pi = np.pi
    convert_factor = -(1.0 / 3.0) * pi * cm2
    cross_sections = np.zeros(rates.shape)
    omega_factors = np.zeros(len(omegas))
    if divide:
        omega_factors = 1.0 / omegas
    else:
        omega_factors = omegas
    # print(rates[:,2])
    i = 0
    for omega_fac in omega_factors:
        if rates.shape != (N,):
            # print(omega, omega*eV_per_Hartree)
            cross_sections[i, :] = omega_fac * rates[i, :] * convert_factor
        else:
            cross_sections[i] = omega_fac * rates[i] * convert_factor
            # print(cross_sections[i], rates[i])

        i += 1
    return cross_sections


def ground_state_energy(data_dir, kappa, n):
    """Loads the ground state energy.
    Arguments:
        - data_dir : str
            The fortran output directory.

        - kappa    : int
            The relativistic quantum number.

        - n        : int
            The principal quantum number.
    """
    hf_file = data_dir + "hf_wavefunctions/hf_energies_kappa_" + str(kappa) + ".dat"

    return [float(l.split()[0]) for l in open(hf_file, "r").readlines()][n - 1]
