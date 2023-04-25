import numpy as np
from matplotlib import pyplot as plt
import sys
# We need to import modules from the relcode_py repository.
# We can add the relcode_py repo to our python path, or we can add it manually in any particular script.
# In this example we do the latter. The path to where the relcode_py repo is located will likely be
# different from what is used in this example!
# If you are using some integrated development environment there is probably a better workflow
# for you for adding the relcode_py repo to your environment.

relcode_py_repo_path = "/home/anton/lx04/relcode_py/"

sys.path.append(relcode_py_repo_path)

from fortran_output_analysis.diagonalisation import Diagonalisation, Resonance
from fortran_output_analysis.constants_and_parameters import g_eV_per_Hartree

# This is an example script that shows usage of the Diagonalisation class and the Resonance class for looking at
# Fortran output data calculated on neon with active holes in 2s and 2p using forward only diagrams.

# The diagonalisation class loads the Fortran diagonalisation data and provides interfaces for manipulating that data.
# For example computing total absorption cross sections or inspecting a particular resonance (via the Resonance class).
neon_diag = Diagonalisation("neon")

data_dir = "/home/anton/lx04/relcode_examples/neon_example/output/"

# There are methods to load each type of output data separately, but this wrapper loads
# channel indices, eigenvalues and "matrix elements" for us, if we provide a valid directory path containing the
# "diag_*.dat" output from Fortran.
neon_diag.load_diag_data_from_directory(data_dir)

# With diagonalisation we can have some arbitrary fine energy grid to compute cross section on
num_omegas = 3000
omega_range_ev = np.linspace(47.0, 54.0, num_omegas)
cross_section = neon_diag.compute_absorption_cross_section(omega_range_ev)

# Example plot of absorption cross section with a line marking 2s1/2 threshold
plt.figure(0)
x = omega_range_ev
y = cross_section
plt.plot(x, y, linewidth=2)
threshold_2s = 1.93583100454*g_eV_per_Hartree
plt.axvline(threshold_2s, linewidth=2, color='black', linestyle='dashed')
plt.title("Absorption cross section for %s near $s_{1/2}$ threshold" % (neon_diag.name))
plt.xlabel("Photon energy [eV]")
plt.ylabel("Cross section [Mbarn]")
plt.grid(linestyle="dotted")


#####
##### Resonance
#####
# How the Fano profile etc are calculated is shown in Lindroth PRA 1995 "Photodetachment of H- and Li-"

# We compute a different more narrow energy grid for looking at the particular resonance.
num_omegas = 3000
omega_range_ev = np.linspace(48.0, 51.0, num_omegas)
omega_range_au = omega_range_ev/g_eV_per_Hartree
cross_section = neon_diag.compute_absorption_cross_section(omega_range_ev)

# It is quite annoying to pick out what eigenvalue corresponds to a particular resonance.
# I usually plot real vs imaginary part of the eigenvalues as below (plot commented out by default).
# Then I can visually see what energy (sort of) that is interesting for any particular resonance I want to look at.
re_eig = np.real(neon_diag.eigenvalues)
imag_eig = np.imag(neon_diag.eigenvalues)
re_eig = re_eig*g_eV_per_Hartree
# Plot Re/Im of eigenvalues to visualise resonance states
# plt.plot(re_eig, imag_eig, 'ko')
# #plt.xlim([48.0, 54.0]) #
# plt.show()

# In this case: first resonance seems to be at 49.7 eV or so (can also be seen from the total cross section plot)
# So we just have some code to pick out the index for those eigenvalues.
interesting_range = np.where(re_eig > 49.5)
possible_resonance_indices = interesting_range[0][:3]
resonance_index = possible_resonance_indices[1]
# This resulted in the following:
print("real(E) = %.8f, imag(E) = %.8f, resonance index=%i" % \
(re_eig[resonance_index], imag_eig[resonance_index], resonance_index))

# When we have an idea of what index the resonant eigenvalue has, we can create an instance of the Resonance class
# using that index and the eigenvalue and matrix element from the Diagonalisation instance.
resonant_eigval = neon_diag.eigenvalues[resonance_index]
resonant_energy = np.real(resonant_eigval)*g_eV_per_Hartree
resonant_matrix_element = neon_diag.matrix_elements[resonance_index]
reso = Resonance(resonance_index, neon_diag.eigenvalues[resonance_index], neon_diag.matrix_elements[resonance_index])

# Either we can choose the q like below, but this condition is also baked into the relevant functions
# so you can skip providing a q_idx if you want, and it will use the condition below for choosing solution.
if reso.I_k > 0:
    q_index = 1  # q1 or q2
else:
    q_index = 0

print("q val = ", reso.q[q_index])

fano_profile = reso.get_fano_profile(omega_range_au)
sigma0 = reso.compute_sigma0()
epsilon = reso.get_epsilon(omega_range_au)

# We can plot the fano profile vs epsilon and investigate if 1/q or -q is min/max respectively.
# We should have picked the q such that epsilon = -q gives the minimum of the fano profile.
plt.figure(1)
plt.plot(epsilon, fano_profile, color="orange", linestyle="dashed", label="fano factor")
plt.axvline(1/reso.q[q_index], linewidth=1, color='gray', linestyle='solid', label="1/q")
plt.axvline(-reso.q[q_index], linewidth=1, color='black', linestyle='dashed', label="-q")
plt.xlim([-6, 6])
plt.title("Fano profile")
plt.xlabel("$\epsilon$")
plt.ylabel("$(q+\epsilon)^2 / (1+\epsilon^2)$")
plt.legend()

# In Lindroth PRA 1995 we see how the cross section should be dominated by the following around
# the resonance:
constant_shift = np.min(cross_section)
sigma_k = sigma0*fano_profile + constant_shift

plt.figure(2)
plt.plot(omega_range_ev, cross_section, color="skyblue", linestyle="solid", label="total abs. cross section")
plt.plot(omega_range_ev, sigma_k, 'kx', label="sigma_k")
plt.axvline(resonant_energy, linewidth=1, color='black', linestyle='dashed')
plt.title("Total absorption cross section vs single resonance cross section sigma_k")
plt.xlabel("Photon energy [eV]")
plt.ylabel("cross section [Mbarn]")
plt.legend()

plt.show()