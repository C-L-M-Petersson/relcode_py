import numpy as np
from matplotlib import pyplot as plt
import sys

# We need to import modules from the relcode_py repository.
# We can add the relcode_py repo to our python path, or we can add it manually in any particular script.
# In this example we do the latter. The path to where the relcode_py repo is located will likely be
# different from what is used in this example!
# If you are using some integrated development environment there is probably a better workflow
# for you for addding the relcode_py repo to your environment.

relcode_py_repo_path = "/home/jsorngard/Mirrors/atomlx04/Repo/relcode_py"

sys.path.append(relcode_py_repo_path)

from fortran_output_analysis.twophotons import TwoPhotons, final_kappas, get_two_photon_asymmetry_parameter, match_matrix_elements_to_same_final_photoelectron_energy
from fortran_output_analysis.common_utility import kappa_from_l_and_j, coulomb_phase
from fortran_output_analysis.constants_and_parameters import g_omega_IR

#
# Defining paths to the Fortran output data:
#

data_dir = "/home/jsorngard/Mirrors/atomlx04/Repo/relcode_examples/argon_example/full/"
twophoton_data_dir = data_dir+"second_photon/"
path_abs = twophoton_data_dir + "m_elements_abs_-2_2.dat"
path_emi = twophoton_data_dir + "m_elements_emi_-2_2.dat"

# Some physical parameters of the system
Z = 1
binding_energy_p1half = 15.9 #eV
binding_energy_p3half = 15.7 #eV

# 
steps_per_IR_photon = 64

# Create instance of the TwoPhotons class.
# The Fortran data can then be added (i.e loaded from file) to this instance.
# The argument to the constructor is just a string that will be stored in two_photons.name.
# It can be used as a plot title or such.
two_photons = TwoPhotons("Ar sidebands 16/18")

#We read in the XUV photon energies here by specifying the path.
two_photons.add_omega(data_dir + "pert_-2_2/omega.dat")
omega = two_photons.omega_eV  # We can also access it like two_photons.omega_Hartree if we want it in a.u.

# Then we can add matrix elements to the TwoPhotons instance.
# We need to give it the path to the output file, and if it's absorption or emission as "abs" or "emi".
# Then we also give it the hole kappa (ie -2 for p3/2 here).
# Last argument is the principal quantum number n, for the hole, mainly used for labels in plots.
hole_kappa = kappa_from_l_and_j(1, 3/2)
two_photons.add_matrix_elements(path_abs, "abs", hole_kappa, 3)  # absorption
two_photons.add_matrix_elements(path_emi, "emi", hole_kappa, 3)  # emission

# We can calculate the possible final kappas with this function
final_kappas = final_kappas(hole_kappa, only_reachable = False)

# Then we can retrieve the value of the coupled matrix elements like so
M_abs_p3half = [two_photons.get_coupled_matrix_element(hole_kappa, "abs", kf) for kf in final_kappas]
M_emi_p3half = [two_photons.get_coupled_matrix_element(hole_kappa, "emi", kf) for kf in final_kappas]

# We compute the final energy of the photoelectron:
ekin_abs = omega - binding_energy_p3half + g_omega_IR
ekin_emi = omega - binding_energy_p3half - g_omega_IR

# We multiply in the coulomb phase of the final photoelectron as well
coul_phase_abs = coulomb_phase(1, ekin_abs, Z)
coul_phase_emi = coulomb_phase(1, ekin_emi, Z)
M_abs_p3half *= np.exp(1j*coul_phase_abs)
M_emi_p3half *= np.exp(1j*coul_phase_emi)

# We then shift the values in the matrix elements so that the same index in
#the absorption and emission arrays correspond to the same final kinetic energy
#for kappa_i in range(5):
#	for K in range(3):
#		#ekin here is the total absorbed photon energy ignoring binding energy
#		ekin_p3half, M_abs_p3half[kappa_i][K], M_emi_p3half[kappa_i][K] = match_matrix_elements_to_same_final_photoelectron_energy(omega, M_abs_p3half[kappa_i][K], M_emi_p3half[kappa_i][K], steps_per_IR_photon)
# This part would normally be run, but since we only have a few datapoints of example data, this would actually remove it all.

# Now we can compute the asymmetry parameters!
b2_p3half_abs, b2_p3half_abs_label = get_two_photon_asymmetry_parameter(2, hole_kappa, M_abs_p3half, M_abs_p3half, "abs")
b4_p3half_abs, b4_p3half_abs_label = get_two_photon_asymmetry_parameter(4, hole_kappa, M_abs_p3half, M_abs_p3half, "abs")

# And show them
plt.plot(omega,b2_p3half_abs,label = b2_p3half_abs_label)
plt.plot(omega,b4_p3half_abs,label = b4_p3half_abs_label)
plt.legend()
plt.xlabel("Photoelectron energy [eV]")
plt.ylabel("Asymmetry parameter value")
plt.show()