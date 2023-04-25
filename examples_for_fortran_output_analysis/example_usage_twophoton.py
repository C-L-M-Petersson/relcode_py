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

from fortran_output_analysis.twophotons import TwoPhotons
from fortran_output_analysis.common_utility import kappa_from_l_and_j

#
# Defining paths to the Fortran output data:
#

data_dir = "/home/jsorngard/Mirrors/atomlx04/Repo/relcode_examples/argon_example/full/"
twophoton_data_dir = data_dir+"second_photon/"

path_abs = twophoton_data_dir + "m_elements_eF2_-2_2.dat"
path_emi = twophoton_data_dir + "m_elements_eF1_-2_2.dat"

# Create instance of the TwoPhotons class.
# The Fortran data can then be added (i.e loaded from file) to this instance.
# The argument to the constructor is just a string that will be stored in two_photons.name.
# It can be used as a plot title or such.
two_photons = TwoPhotons("Ar sidebands 16/18")


# We read in the XUV photon energies here by specifying the path.
two_photons.add_omega(data_dir + "pert_-2_2/omega.dat")
omega = two_photons.omega_eV  # We can also access it like two_photons.omega_Hartree if we want it in a.u.

# Then we can add matrix elements to the TwoPhotons instance.
# We need to give it the path to the output file, and if it's absorption or emission as "abs" or "emi".
# Then we also give it the hole kappa (ie -2 for p3/2 here).
# Last argument is the principal quantum number n, for the hole, mainly used for labels in plots.
two_photons.add_matrix_elements(path_abs, "abs", -2, 3)  # absorption
two_photons.add_matrix_elements(path_emi, "emi", -2, 3)  # emission

# There is also a helper function to translate from l,j to kappa:
print("l=1, j=3/2 -> kappa: ", kappa_from_l_and_j(l=1, j=3/2))  # Should be -2 for l=1, j=3/2

# To add the output from a different hole, in this case 3p1/2 we change the paths and kappa accordingly.
# Note that we overwrite the path variables used previously, this is just a matter of convenience.
path_abs = twophoton_data_dir + "m_elements_eF2_1_2.dat"
path_emi = twophoton_data_dir + "m_elements_eF1_1_2.dat"

# We can use the helper functions if we don't know the maps (l,j) <-> kappa by heart
kappa_3p1half = kappa_from_l_and_j(l=1, j=1/2)
print("l=1, j=1/2 -> kappa: ", kappa_3p1half)  # Should be 1 for l=1, j=1/2
n = 3
two_photons.add_matrix_elements(path_abs, "abs", kappa_3p1half, n)  # absorption
two_photons.add_matrix_elements(path_emi, "emi", kappa_3p1half, n)  # emission

# When we have added matrix elements to the instance, we can retrieve the data.
# Either we get the raw data from a specifc column in the Fortran output file, by specifying the
# relevant kappas for (hole, intermediate, final).
# Example (-2, -1, 1) - 3p3/2 -> s1/2 -> p1/2
hole_kappa = -2
intermediate_kappa = -1
final_kappa = 1
# We get back a complex valued numpy array with the data for (in this case) absorption two-photon matrix element.
# Here we name this return value z, it can be whatever. We also get back a string stored in channel_name.
z, channel_name = two_photons.get_matrix_element_for_intermediate_resolved_channel(hole_kappa,
                                                                                   intermediate_kappa,
                                                                                   final_kappa,
                                                                                   "abs")
# The string can be used for plotting labels ie, it looks like follows:
print("string corresponding to matrix element for (hole, intermediate, final):", channel_name, "\n")

# Note that when we get matrix elements for a particular channel (hole, intermediate, final), we
# don't take any consideration to mj for describing the states.

# We can also get matrix elements summed over the intermediate channels. Then we have
# to specify an mj-value since we will be calculating 3j-symbols (internally) when getting these matrix elements.
# The TwoPhotons class automatically takes care of forbidden channels depening on the choice of (hole, final) and
# mj.
# We can get a list of the possible channels like so, for an example of 3p3/2, mj=1/2 to all final states (for emission say):
abs_or_emi = "emi"
mj = 0.5
hole_kappa = -2
channels_3p3half = two_photons.get_hole_to_final_channels(hole_kappa, abs_or_emi, mj)
print("Channels from hole 3p3/2 to final continuum states, "
      "for mj=1/2, are described by these (hole_kappa, final_kappa pairs): ")
print(channels_3p3half, "\n")
# Changing mj=1.5 should remove all states ending up in p1/2 for example.

# Getting the channels is just a helper function to get all the possible channels.
# We can use this to get the summed matrix elements, for a particular mj, like so:
for channel in channels_3p3half:
    final_kappa = channel[1]  # Getting the second element of a (hole_kappa, final_kappa)-tuple.
    # Here we store the data for a two-photon matrix element in variable called z.
    # It is the data after summing over all intermediate states including 3j-symbols for a particular mj.
    # Note that each hole/final state is fully specified by a kappa and an mj-value!
    z, name = two_photons.get_matrix_elements_for_hole_to_final(hole_kappa, final_kappa, abs_or_emi, mj)
    # These can be used to form products of matrix elements for absorption/emission, and then the sum over
    # mj can be performed etc.
    # The name-variable has some information to be used in plotting labels as well:
    print("For (hole_kappa, final_kappa) with mj = %i/2:" % (int(2*mj)))
    print(channel, "$"+name+"$", "\n")

    plt.plot(omega, np.real(z))

plt.show()
