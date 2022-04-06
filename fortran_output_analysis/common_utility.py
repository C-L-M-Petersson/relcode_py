# ==================================================================================================
# This file contains helper functions that are shared among several parts of the
# atomicsystem-scripts.
# Examples are kappa <-> l,j and wigner 3j-symbol functions
# ==================================================================================================
import numpy as np
from sympy import N as sympy_to_num
from sympy.physics.wigner import wigner_3j

# ==================================================================================================
#
# ==================================================================================================
class IonHole:
    def __init__(self, kappa, n_qn):
        self.kappa = kappa
        self.n = n_qn  # n quantum number (principal)
        self.l = l_from_kappa(kappa)
        self.j = j_from_kappa(kappa)
        self.name = str(self.n) + l_to_str(self.l) + ("_{%i/2}" % (j_from_kappa_int(kappa)))

# ==================================================================================================
#
# ==================================================================================================
def load_raw_data(path):
    return np.loadtxt(path)


def kappa_from_l_and_j(l, j):
    if (int(2 * l) == int(2 * j) - 1):
        return -(l+1)
    else:
        return l


def l_from_kappa(kappa):
    if (kappa < 0):
        return -kappa - 1
    else:
        return kappa


def j_from_kappa(kappa):
    l = l_from_kappa(kappa)
    if (kappa < 0):
        return l + 0.5
    else:
        return l - 0.5


def j_from_kappa_int(kappa):
    l = l_from_kappa(kappa)
    if (kappa < 0):
        return 2 * l + 1
    else:
        return 2 * l - 1


def l_from_str(l_str):
    l = -1

    if (l_str == "s"):
        l = 0
    elif (l_str == "p"):
        l = 1
    elif (l_str == "d"):
        l = 2
    elif (l_str == "f"):
        l = 3

    if (l == -1):
        raise ValueError("l_from_str(): invalid or unimplemented string for l quantum number.")
    else:
        return l


def l_to_str(l):
    if (l == 0):
        return "s"
    elif (l == 1):
        return "p"
    elif (l == 2):
        return "d"
    elif (l == 3):
        return "f"
    else:
        raise ValueError("l_to_str(): invalid or unimplemented string for l quantum number.")


# ==================================================================================================
#
# ==================================================================================================
def wigner3j_numerical(hole_kappa, final_kappa, mj):
    mjj = int(2*mj)
    j_hole = j_from_kappa_int(hole_kappa)
    j_final = j_from_kappa_int(final_kappa)
    if(mjj > j_hole or mjj > j_final):
        return
    #print("j_hole, j_final, mj: %i/2 %i/2 %i/2" % (j_hole, j_final, mjj))
    K = 1
    q = 0
    w3j = wigner_3j(j_final/2, K, j_hole/2, -mjj/2, q, mjj/2)
    #print(w3j, sympy_to_num(w3j))
    return sympy_to_num(w3j)

def wigner3j_numerical2(j_hole, j_final, mjj):
    #print("j_hole, j_final, mj: %i/2 %i/2 %i/2" % (j_hole, j_final, mjj))
    K = 1
    q = 0
    w3j = wigner_3j(j_final/2, K, j_hole/2, -mjj/2, q, mjj/2)
    #print(w3j, sympy_to_num(w3j))
    return sympy_to_num(w3j)

def wigner_eckart_phase(final_kappa, mj):
    return np.power(-1.0, (j_from_kappa(final_kappa)-mj))

# ==================================================================================================
#
# ==================================================================================================
def convert_rate_to_cross_section(rates, omegas, divide=True):
    N = len(omegas)
    cm2 = (0.52917721092 ** 2) * 100.0
    pi = np.pi
    convert_factor = -(1.0 / 3.0) * pi * cm2
    cross_sections = np.zeros(rates.shape)
    omega_factors = np.zeros(len(omegas))
    if(divide):
        omega_factors = 1.0/omegas
    else:
        omega_factors = omegas
    #print(rates[:,2])
    i = 0
    for omega_fac in omega_factors:
        if(rates.shape != (N,)):
        # print(omega, omega*eV_per_Hartree)
            cross_sections[i, :] = omega_fac * rates[i, :] * convert_factor
        else:
            cross_sections[i] = omega_fac * rates[i] * convert_factor
            #print(cross_sections[i], rates[i])

        i+=1
    return cross_sections