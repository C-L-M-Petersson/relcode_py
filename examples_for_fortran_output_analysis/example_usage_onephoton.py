import numpy as np
from matplotlib import pyplot as plt
import sys
# We need to import modules from the relcode_py repository.
# We can add the relcode_py repo to our python path, or we can add it manually in any particular script.
# In this example we do the latter. The path to where the relcode_py repo is located will likely be
# different from what is used in this example!
# If you are using some integrated development environment there is probably a better workflow
# for you for addding the relcode_py repo to your environment.

relcode_py_repo_path = "/home/anton/lx04/relcode_py/"

sys.path.append(relcode_py_repo_path)

from fortran_output_analysis.common_utility import kappa_from_l_and_j
from fortran_output_analysis.onephoton import OnePhoton

data_dir = "/home/anton/lx04/relcode_examples/neon_example/output/"

# We create an instance of the OnePhoton class as follows, providing a name for the atom.
# In this example we take some data from neon calculations close to the
# 2s^-1 3p Fano resonance leading up to the 2s threshold.
one_photon = OnePhoton("Neon 2s3p")

# If we want to look at, for example, the one-photon cross section,
# we can add a set of ionisation channels for a particular hole.
# The Fortran program outputs the probability current for ionisation from a hole to the set of possible final states
# in the continuum. So we add all these "channels" for a particular hole (described by a kappa).
# In our example we look at ionisation from 2p1/2 and 2p3/2 (kappa 1 and kappa -2 respectively).

# Add channels from 2p3/2
hole_kappa = -2
n_principal_quantum_number = 2
# Specify path to the probability currents
path_to_pcur_all = data_dir+"pert_-2_1/pcur_all.dat"
# Current interface to add channels from a hole looks like this:
# (principal quantum number is mainly for generating labels for plots).
one_photon.add_channels_for_hole(path_to_pcur_all, hole_kappa, n_principal_quantum_number)
# This loads the Fortran output data for the probability current into the OnePhoton instance,
# and it sets up what the possible ionisation channels are.
# We can get the labels for them as a list f.ex:
labels_from_2p3half = one_photon.get_channel_labels_for_hole(hole_kappa)
# These can be used for plotting.
print(labels_from_2p3half)

# Usually we want to look at the photoionisation cross section after absorption of the XUV photon.
# This is calculated from the probability current ("ionisation rate").
# There are methods that calculate the partial cross sections:
final_kappa = -1  # kappa=-1 -> l=0, j=1/2, ie final state s1/2
cross_section_2p3half_to_s1half = one_photon.get_partial_cross_section(hole_kappa, final_kappa)  # This is in Mbarn

# Here we get the partial cross section for 2p3/2 -> d5/2 using a helper function to get the kappa.
cross_section_2p3half_to_d5half = one_photon.get_partial_cross_section(hole_kappa, kappa_from_l_and_j(2, 5/2))

# While we easily can get all available partial cross sections and sum them, the code for doing this is already in
# the OnePhoton class.
# Let's also add a new hole, 2p1/2:
one_photon.add_channels_for_hole(data_dir+"pert_1_1/pcur_all.dat", 1, 2)
# The get_total_cross_section()-method will sum up the cross sections for all added channels.
total_cross_section = one_photon.get_total_cross_section()

# We can get the list of photon energies (of which the cross sections are a function of).
# In eV or atomic units:
omega_XUV_eV = one_photon.get_omega_eV()
omega_XUV_au = one_photon.get_omega_Hartree()

plt.plot(omega_XUV_eV, cross_section_2p3half_to_s1half, label=labels_from_2p3half[0])
plt.plot(omega_XUV_eV, cross_section_2p3half_to_d5half, label=labels_from_2p3half[2])
plt.plot(omega_XUV_eV, total_cross_section, label="Total cross section from 2p3/2 and 2p1/2")
plt.legend()
plt.show()