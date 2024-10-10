import sys
import os
import numpy as np
from matplotlib import pyplot as plt

"""
Sometimes we may want to compute Wigner delay and phase based on the data from two different
simultations: one for absorption path and one for emission path. In this scenario, we have to
properly match matrices from both simulations and compute the phase/delay on the interpolated
values (matrix elements). 

The logic for handling this is included in all the functions in the onephoton_delays_and_phases
namespace. You just need to provide the second OnePhoton object (corresponding to the second 
simulation) and a couple of additional parameters which are described below.

We'll show how it's supposed to work in the case of ionization of 5p_3/2 electron in Xenon.
NOTE: all the detailed explanation for the case of one simulation only is shown in 
example_onephoton_matching.py
"""

# append relcode_py to our system path to easily import all the necessary modules
relcode_py_repo_path = "D:\\relcode_py"
sys.path.append(relcode_py_repo_path)

# initialize two one_photon objects, the hole and load this hole in both
from fortran_output_analysis.onephoton.onephoton import OnePhoton
from fortran_output_analysis.common_utility import Hole
from fortran_output_analysis.constants_and_parameters import g_eV_per_Hartree

atom_name = "Xenon"
g_omega_IR = (
    1.55 / g_eV_per_Hartree
)  # energy of IR photon used in simulations (in Hartree)

one_photon_emi = OnePhoton(
    atom_name, g_omega_IR
)  # object for the first simulation (emission path)
one_photon_abs = OnePhoton(
    atom_name, g_omega_IR
)  # object for the second simulation (absorption path)


# initialize 5p_3/2 hole
hole_kappa_5p3half = -2
hole_n_5p3half = 5

# specify path to Fortran output data (I use backslash for the path since I work on Windows)
data_dir_emi = (
    "fortran_data\\xenon_emission\\"  # data for the first simulation (emission path)
)
data_dir_abs = "fortran_data\\xenon_absorption\\"  # data for the second simulation (absorption path)

# load the hole
one_photon_emi.load_hole(hole_n_5p3half, hole_kappa_5p3half, data_dir_emi)
one_photon_abs.load_hole(hole_n_5p3half, hole_kappa_5p3half, data_dir_abs)

# get Wigner phase and delay
import fortran_output_analysis.onephoton.onephoton_delays_and_phases as wigner

angles = np.array([0, 30, 60])  # angles to compute the total delay/phase for
Z = 1  # charge of the ion

plt.figure("Total wigner delay")
for angle in angles:
    en, delay = wigner.get_wigner_delay(
        one_photon_emi,
        hole_n_5p3half,
        hole_kappa_5p3half,
        Z,
        angle,
        one_photon_2=one_photon_abs,
        energies_mode="both",
    )
    plt.plot(en, delay, label=f"{angle}")

plt.legend()
plt.title("Total wigner delay for 5p_3/2")


plt.figure("Total wigner phase")
for angle in angles:
    en, phase = wigner.get_wigner_phase(
        one_photon_emi,
        hole_n_5p3half,
        hole_kappa_5p3half,
        Z,
        angle,
        one_photon_2=one_photon_abs,
        energies_mode="both",
    )
    plt.plot(en, phase, label=f"{angle}")

plt.legend()
plt.title("Total wigner phase for 5p_3/2")

plt.show()
input()

"""
The energies_mode parameter in the functions above specifies the array of energies that we
use to match our simulations. Poissble options:
"both" (default) - concatenate energy arrays from both simulations
"abs" - take the energy array from the simulation for absorption path
"emi" - take the energy array from the simulation for emission path
"""
