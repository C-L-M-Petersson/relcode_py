import sys
import numpy as np
from matplotlib import pyplot as plt

"""
This code will help you understand how to use OnePhoton class and different onephoton_...  
namespaces to proccess Fortran output data for the one photon case.

At first, we need to import modules from the relcode_py repository. We have two possible ways
to do this:
1. Add __init__.py file to this folder as well as all the folders from which we import code 
and then run this script using "python" command with "-m" flag (run module as a script). 
The full command is: "python -m examples_for_fortran_output_analysis.example_usage_onephoton".
2. Add relcode_py repository to our system path through sys.path.append("path/to/relcode_py")
and run this script as usual.

In this example, we'll use the second approach.
"""

relcode_py_repo_path = "D:\\relcode_py"
# append relcode_py to our system path
sys.path.append(relcode_py_repo_path)

# now we can easily import our OnePhoton class
from fortran_output_analysis.onephoton.onephoton import OnePhoton
from fortran_output_analysis.common_utility import Hole

# we also import some physical constants required for the analysis
from fortran_output_analysis.constants_and_parameters import g_eV_per_Hartree


# ============== Initialization with OnePhoton ==============
"""
The main purpose of the OnePhoton class is to initialize the atom, the holes we want to 
consider ionization from and load the raw Fortran output data. 

In this guide we'll consider ionization from the Radon's 6p_3/2 and 6p_1/2 holes. 
"""

# create an instance
g_omega_IR = (
    1.55 / g_eV_per_Hartree
)  # energy of IR photon used in simulations (in Hartree)
one_photon = OnePhoton("Radon", g_omega_IR)

# specify path to Fortran output data (I use backslash for the path since I work on Windows)
data_dir = "fortran_data\\2_-4_64_radon\\"

"""
To load diagonal matrix elements and diagonal eigenvalues (required for e.g. photoabsorption 
cross section) we use "load_diag_data" method inside the one_photon object. The method contains 
"should_reload" parameter that tells wheter we should reinitalize diagonal data if they were
previously loaded (False by default).

Then we need to initialize some hole objects and load raw data for them. 
To load a hole we call a method named "load_hole" inside the one_photon object. 
The Fortran program outputs the probability current for ionisation from a hole to the set of 
possible final states in the continuum (channels). So, when we load a hole we also add all these 
"channels". "load_hole" method also contains "should_reload" parameter that tells wheter we 
should reinitalize a hole if that hole was previously loaded (False by default).

The data for loaded holes are stored in the self.channels dictionary attribute of one_photon object 
and can be accessed via the "get_channel_for_hole" method.
"""

# load diagonal data
one_photon.load_diag_data(data_dir)

# try to reload diagonal data (outputs information message)
one_photon.load_diag_data(data_dir, should_reload=True)

# initialize 6p_3/2 hole
hole_kappa_6p3half = -2
hole_n_6p3half = 6
binding_energy_6p3half = 0.395  # (in Hartree)
hole_6p3half = Hole(hole_kappa_6p3half, hole_n_6p3half, binding_energy_6p3half)

# load 6p_3/2
path_to_pcur_6p3half = data_dir + "pert_-2_5\pcur_all.dat"
path_to_omega_6p3half = data_dir + "pert_-2_5\omega.dat"
one_photon.load_hole(
    path_to_pcur_6p3half,
    hole_6p3half,
)  # initialize all possible ionization channels and loads raw data for them

# We can get the labels for all possible inonization channels from 6p_3/2 hole:
labels_from_6p3half = one_photon.get_channel_labels_for_hole(hole_6p3half)
print(f"Possible channels for 6p_3/2 hole: {labels_from_6p3half}")

# initialize 6p_1/2 hole
hole_kappa_6p1half = 1
hole_n_6p1half = 6
binding_energy_6p1half = 0.53578  # (in Hartree)
hole_6p1half = Hole(hole_kappa_6p1half, hole_n_6p1half, binding_energy_6p1half)

# load 6p_1/2 hole
path_to_pcur_6p1half = data_dir + "pert_1_5\pcur_all.dat"
path_to_omega_6p1half = data_dir + "pert_1_5\omega.dat"
one_photon.load_hole(
    path_to_pcur_6p1half,
    hole_6p1half,
)  # initialize all possible ionization channels and loads raw data for them

# We can get the labels for all possible inonization channels from 6p_1/2 hole:
labels_from_6p1half = one_photon.get_channel_labels_for_hole(hole_6p1half)
print(f"Possible channels for 6p_1/2 hole: {labels_from_6p1half}")

# try to reload 6p_1/2 hole with the same data (outputs information message)
one_photon.load_hole(
    path_to_pcur_6p1half,
    hole_6p1half,
    should_reload=True,
)

# ============== Analysis with onephoton_... namespaces ==============
"""
onephoton folder contains different namespaces with functions that analyse raw output Fortran 
data and obtain meaningful physical properties. There is a namespace for cross sections, 
a namespace for asymmetry parameters, etc.

Almost all functions in these namespaces require an object of OnePhoton class with some 
loaded holes as input.
"""

# 1. Photon and photoelctron kinetic energies with onephoton_utilities.
"""
It's usually important to check for which XUV photon/photoelectron energies our data were computed 
in the Fortran simulations. We can easily do this by calling the methods from 
onephoton_utilities namespace shown below:
"""
import fortran_output_analysis.onephoton.onephoton_utilities as util

# energies for 6p_3/2 (similarly for 6p_1/2)

# XUV photon energies in eV
en = util.get_omega_eV(one_photon, hole_6p3half)

# XUV photon energies in Hartree
en = util.get_omega_Hartree(one_photon, hole_6p3half)

# photoelectron kinetic energies in eV
en = util.get_electron_kinetic_energy_eV(one_photon, hole_6p3half)

# photoelectron kinetic energies in Hartree
en = util.get_electron_kinetic_energy_Hartree(one_photon, hole_6p3half)

# 2. Integrated cross sections with onephoton_cross_sections
"""
Usually we want to look at the integrated (over all angles) photoionisation cross sections
after absorption of the XUV photon. All the required methods are contained in 
onephoton_cross_sections namespace.
We can calculate cross sections from probability current using mode="pcur" parameter in the 
functions and from matrix amplitudes using mode="amp".

NOTE: All these methods for cross sections below also return the corresponding
photoelectron kinetic energies in eV.
"""
import fortran_output_analysis.onephoton.onephoton_cross_sections as cs

# partial integrated cross section for the ionziation 6p_3/2 -> d_3/2
# calculated in two ways: prob current and matrix amplitudes
kappa_d3half = 2
en, cs_pcur = cs.get_partial_integrated_cross_section_1_channel(
    one_photon, hole_6p3half, kappa_d3half, mode="pcur"
)
en, cs_amp = cs.get_partial_integrated_cross_section_1_channel(
    one_photon, hole_6p3half, kappa_d3half, mode="amp"
)

plt.figure("Partial integrated crossection for 6p_3/2 -> d_3/2 ")
plt.plot(en, cs_pcur, label="pcur")
plt.plot(en, cs_amp, "--", label="amp")
plt.legend()
plt.title("Partial integrated crossection for 6p_3/2 -> d_3/2 ")

# partial integrated cross section for two channels: 6p_3/2 -> d_3/2 and 6p_3/2 -> d_5/2
# calculated in two ways: prob current and matrix amplitudes
kappa_d5half = -3
final_kappas = [kappa_d3half, kappa_d5half]
en, cs_pcur = cs.get_partial_integrated_cross_section_multiple_channels(
    one_photon, hole_6p3half, final_kappas, mode="pcur"
)
en, cs_amp = cs.get_partial_integrated_cross_section_multiple_channels(
    one_photon, hole_6p3half, final_kappas, mode="amp"
)
plt.figure(
    "Partial integrated crossection for two channels: 6p_3/2 -> d_3/2 and 6p_3/2 -> d_5/2"
)
plt.plot(en, cs_pcur, label="pcur")
plt.plot(en, cs_amp, "--", label="amp")
plt.legend()
plt.title(
    "Partial integrated crossection for two channels: 6p_3/2 -> d_3/2 and 6p_3/2 -> d_5/2"
)

# total integrated cross section for 6p_3/2
# calculated in two ways: prob current and matrix amplitudes
en, cs_pcur = cs.get_total_integrated_cross_section_for_hole(
    one_photon, hole_6p3half, mode="pcur"
)
en, cs_amp = cs.get_total_integrated_cross_section_for_hole(
    one_photon, hole_6p3half, mode="amp"
)
plt.figure("Total integrated crossection for hole 6p_3/2")
plt.plot(en, cs_pcur, label="pcur")
plt.plot(en, cs_amp, "--", label="amp")
plt.title(f"Total integrated crossection for hole 6p_3/2")

# Integrated photoelectron emission cross section. Can be computed in two energy modes:
# 1. "omega" mode when we just compute the sum of cross sections for all loaded holes
# and return the result for photon energies.
# 2. "ekin" mode when we compute cross sections for electron kinetic energies (which are
# different for different holes) and then interpolate them so that they match the same final
# kinetic energies.

# "omega" mode
# calculated in two ways: prob current and matrix amplitudes
omega, cs_pcur = cs.get_integrated_photoelectron_emission_cross_section(
    one_photon, mode_energies="omega", mode_cs="pcur"
)
omega, cs_amp = cs.get_integrated_photoelectron_emission_cross_section(
    one_photon, mode_energies="omega", mode_cs="amp"
)
plt.figure("Integrated photonelectron emission cross section for photon energies")
plt.plot(omega, cs_pcur, label="pcur")
plt.plot(omega, cs_amp, "--", label="amp")
plt.legend()
plt.title(f"Integrated photonelectron emission cross section for photon energies")

# "ekin" mode
# calculated in two ways: prob current and matrix amplitudes
ekin, cs_pcur = cs.get_integrated_photoelectron_emission_cross_section(
    one_photon, mode_energies="ekin", mode_cs="pcur"
)

ekin, cs_amp = cs.get_integrated_photoelectron_emission_cross_section(
    one_photon, mode_energies="ekin", mode_cs="amp"
)

plt.figure(
    "Integrated photonelectron emission cross section for photoelectron energies"
)
plt.plot(ekin, cs_pcur, label="pcur")
plt.plot(ekin, cs_amp, "--", label="amp")
plt.legend()
plt.title("Integrated photonelectron emission cross section for photoelectron energies")

# 3. Photoabsorption cross section with onephoton_cross_sections
"""
Photoabsorption cross section is slightly different from the cross sections shown above.
We need diagonal data (eigenvalues and matrix elements) for computations, so be sure that they
are loaded in advance (check the initialization section). We aslo must provide array
of photon energies (in eV) for which we want to compute the cross section.
"""

# Photoabsorption cross section. Requires diagonal eigenvalues and matrix elements to be loaded.
omega = util.get_omega_eV(one_photon, hole_6p3half)
cs_photoabs = cs.get_photoabsorption_cross_section(one_photon, omega)

plt.figure("Photoabsorption cross section")
plt.plot(omega, cs_photoabs)
plt.title("Photoabsorption cross section")

# 4. Angular part of a hole's cross section
"""
For in depth analysis of ionization from a hole, we should also consider angular part of its 
total cross section. The angular part is computed thorugh the real asymmetry parameter. 
The methods for asymmetry parameters are contained in onephoton_asymmetry_parameter namespace.

NOTE: All methods also return corresponding photoelectron energy in eV.
"""
import fortran_output_analysis.onephoton.onephoton_asymmetry_parameters as asym_p

# specify angles to compute angular part of the hole's cross section
angles = np.array([0, 30, 45, 60])

# We also need ion charge which is usually just 1
Z = 1

# compute real asymmetry parameter
en, b2_real = asym_p.get_real_asymmetry_parameter(one_photon, hole_6p3half, Z)
plt.figure("Real asymmetry parameter")
plt.plot(en, b2_real)
plt.title("Real asymmetry parameter")

# angular part of the cross section for 6p_3/2
plt.figure("Angular part of cross section for 6p_3/2")
for angle in angles:
    ekin, ang_cs = cs.get_angular_part_of_cross_section(
        one_photon, hole_6p3half, Z, angle
    )
    plt.plot(ekin, ang_cs, label=f"{angle}")
plt.title("Angular part of cross section for 6p_3/2")
plt.legend()

# total cross section for 6p_3/2 (angular + integrated parts)
plt.figure("Total cross section for 6p_3/2")
for angle in angles:
    ekin, ang_cs = cs.get_total_cross_section_for_hole(
        one_photon, hole_6p3half, Z, angle
    )
    plt.plot(ekin, ang_cs, label=f"{angle}")
plt.title("Total cross section for 6p_3/2")
plt.legend()

# 5. Wigner delay and phases
"""
The last but not least property we usually want to investigate in the one photon case is the
Wigner delay (phase). And we actually want to consider both: integrated and angular parts.
The integrated part is computed from the so-called "Wigner intensity". The angular part is computed
through the complex asymmetry parameter. onephoton_delays_and_phases namespace includes all 
the necessary methods for such computations for both delay and phase. 
The usage examples are shown below.

NOTE: all the method below also return corresponding photoelectron kinetic energies in eV.
"""
import fortran_output_analysis.onephoton.onephoton_delays_and_phases as wigner

# specify angles to compute angular part of the Wigner delay
angles = np.array([0, 30, 45, 60])

# We also need ion charge which is usually just 1
Z = 1

# complex asymmetry parameter
en, b2_complex = asym_p.get_complex_asymmetry_parameter(
    one_photon,
    hole_6p3half,
    Z,
)
# integrated Wigner delay for 6p_3/2
en, delay = wigner.get_integrated_wigner_delay(
    one_photon,
    hole_6p3half,
    Z,
)
plt.figure("Integrated Wigner delay for 6p_3/2")
plt.plot(en, delay)
plt.title("Integrated Wigner delay for 6p_3/2")

# integrated Wigner phase for 6p_3/2
en, phase = wigner.get_integrated_wigner_phase(
    one_photon,
    hole_6p3half,
    Z,
)
plt.figure("Integrated Wigner phase for 6p_3/2")
plt.plot(en, phase)
plt.title("Integrated Wigner phase for 6p_3/2")

# angular part of Wigner delay for 6p_3/2
plt.figure("Angular part of Wigner delay for 6p_3/2")
for angle in angles:
    en, delay = wigner.get_angular_wigner_delay(
        one_photon,
        hole_6p3half,
        Z,
        angle,
    )
    plt.plot(en, delay, label=f"{angle}")
plt.legend()
plt.title("Angular part of Wigner delay for 6p_3/2")

# angular part of Wigner phase for 6p_3/2
plt.figure("Angular part of Wigner phase for 6p_3/2")
for angle in angles:
    en, phase = wigner.get_angular_wigner_phase(
        one_photon,
        hole_6p3half,
        Z,
        angle,
    )
    plt.plot(en, phase, label=f"{angle}")
plt.legend()
plt.title("Angular part of Wigner phase for 6p_3/2")

# Total (integrated + angular) Wigner delay for 6p_3/2
plt.figure("Total wigner delay")
for angle in angles:
    en, delay = wigner.get_wigner_delay(
        one_photon,
        hole_6p3half,
        Z,
        angle,
    )
    plt.plot(en, delay, label=f"{angle}")
plt.legend()
plt.title("Total wigner delay for 6p_3/2")

# Total (integrated + angular) Wigner phase for 6p_3/2
plt.figure("Total wigner phase")
for angle in angles:
    en, phase = wigner.get_wigner_phase(
        one_photon,
        hole_6p3half,
        Z,
        angle,
    )
    plt.plot(en, phase, label=f"{angle}")
plt.legend()
plt.title("Total wigner phase for 6p_3/2")

"""
Optional: "steps_per_IR_photon" gives the XUV step size (as g_omega_IR/steps_per_IR_photon).
NOTE(Leon): What does this mean in the context of non-linear energy grids?

If no value is given, it is calculated from "omegas.dat".
"""
plt.show()
input()

# 6. Wigner delay and phases for two simulations
"""
Sometimes we may want to compute Wigner delay and phase based on the data from two different 
simultations: one for absorption path and one for emission path. In this scenario, we have to
properly match matrices from both simulations and compute the phase/delay on the interpolated 
values (matrix elements). The logic for handling this is included in all the functions shown 
under section "5. Wigner delay and phases". You just need to provide the second OnePhoton object 
(corresponding to the second simulation) and a couple of additional parameters which are described
below.
"""
# TODO: add real examples of including the second simulation and proper description of new parameters
