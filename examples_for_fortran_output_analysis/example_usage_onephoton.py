import sys
import numpy as np
from matplotlib import pyplot as plt

"""
This code will help you understand how to use OnePhoton and OnePhotonAnalyzer classes to proccess 
Fortran output data for the one photon case.

At first, we need to import modules from the relcode_py repository. We have two possible ways
to do this:
1. Add __init__.py file to this folder as well as all the folders from which we import code 
and then run this script using "python" command with "-m" flag (run module as a script). 
The full command is: "python -m examples_for_fortran_output_analysis.example_usage_onephoton".
2. Add relcode_py repository to our system path through sys.path.append("path/to/relcode_py") and
run this script as usual.

In this example, we'll use the second approach.
"""

# path to the relcode_py repository. NOTE, for your system it'll likely be different!!
relcode_py_repo_path = "D:\\relcode_py"
# append relcode_py to our system path
sys.path.append(relcode_py_repo_path)

# now we can easily import our OnePhoton and OnePhotonAnalyzer classes
from fortran_output_analysis.onephoton.onephoton import OnePhoton
from fortran_output_analysis.onephoton.onephoton_analyzer import OnePhotonAnalyzer

# we also import some physical constants required for the analysis
from fortran_output_analysis.constants_and_parameters import g_eV_per_Hartree


# ============== Initialization with OnePhoton ==============
"""
The main purpose of the OnePhoton class is to initialize the atom, the holes we want to consider 
ionization from and load the raw Fortran output data for these holes. 

In this guide we'll consider ionization from the Radon's 6p_3/2 and 6p_1/2 holes. 
"""

# we create an instance of the OnePhoton class as follows, providing a name for the atom.
one_photon = OnePhoton("Radon")

# specify path to Fortran output data (I use backslash for the path since I work on Windows)
data_dir = "fortran_data\\2_-4_64_radon\\"  # NOTE: change this to relevant path

"""
To initialize a hole we call a method named "intialize_hole" inside the one_photon object. For this call
we need several input arguments, namely:
path_to_pcur - path to the file that contains probability current for ionisation from the hole
path_to_omega - path to the corresponding photon energy data (in Hartree)
hole_kappa - kappa value of the hole
hole_n - principal quantum number of the hole
binding_energy - binding energy of the hole
g_omega_IR - the energy of IR photon used in Fortran simulations (in Hartree)

The Fortran program outputs the probability current for ionisation from a hole to the set of 
possible final states in the continuum (channels). So, when we initialize a hole we also add all these 
"channels". In our example we look at ionisations from 6p_3/2 and 6p_1/2 Radon's holes 
(kappa -2 and kappa 1 respectively).
"""

# initialization of the 6p_3/2 hole
path_to_pcur_6p3half = data_dir + "pert_-2_5\pcur_all.dat"
path_to_omega_6p3half = data_dir + "pert_-2_5\omega.dat"
hole_kappa_6p3half = -2
hole_n_6p3half = 6
binding_energy_6p3half = 0.395  # (in Hartree)
g_omega_IR_6p3half = 1.55 / g_eV_per_Hartree  # (in Hartree)
one_photon.initialize_hole(
    path_to_pcur_6p3half,
    hole_kappa_6p3half,
    hole_n_6p3half,
    binding_energy_6p3half,
    g_omega_IR_6p3half,
)  # initialize hole's parameters and loads raw data for all possible ionization channels

# We can get the labels for all possible inonization channels from 6p_3/2 hole:
labels_from_6p3half = one_photon.get_channel_labels_for_hole(hole_kappa_6p3half)
print(f"Possible channels for 6p_3/2 hole: {labels_from_6p3half}")

# initialization of the 6p_1/2 hole
path_to_pcur_6p1half = data_dir + "pert_1_5\pcur_all.dat"
path_to_omega_6p1half = data_dir + "pert_1_5\omega.dat"
hole_kappa_6p1half = 1
hole_n_6p1half = 6
binding_energy_6p1half = 0.53578  # (in Hartree)
g_omega_IR_6p1half = 1.55 / g_eV_per_Hartree  # (in Hartree)
one_photon.initialize_hole(
    path_to_pcur_6p1half,
    hole_kappa_6p1half,
    hole_n_6p1half,
    binding_energy_6p1half,
    g_omega_IR_6p1half,
)  # initialize hole's parameters and loads raw data for all possible ionization channels

# We can get the labels for all possible inonization channels from 6p_1/2 hole:
labels_from_6p1half = one_photon.get_channel_labels_for_hole(hole_kappa_6p1half)
print(f"Possible channels for 6p_1/2 hole: {labels_from_6p1half}")

"""
"initialize_hole" method can also handle reinitialization of the previously initialized hole. If you
pass hole_kappa that already exists in the one_photon object, you will be asked whether you wish to
reinitialize the hole with the new data. You should type in terminal: "Yes" or "No".

The code below shows the reinitialization logic in action (for 6p_1/2 hole). It's just a toy example 
for demonstration since we try to reinitialize the hole with the same data, which has no sense.
"""
# try to reinitialize 6p_1/2 hole wit the same data
path_to_pcur_6p1half = data_dir + "pert_1_5\pcur_all.dat"
path_to_omega_6p1half = data_dir + "pert_1_5\omega.dat"
hole_kappa_6p1half = 1
hole_n_6p1half = 6
binding_energy_6p1half = 0.53578  # (in Hartree)
g_omega_IR_6p1half = 1.55 / g_eV_per_Hartree  # (in Hartree)
one_photon.initialize_hole(
    path_to_pcur_6p1half,
    hole_kappa_6p1half,
    hole_n_6p1half,
    binding_energy_6p1half,
    g_omega_IR_6p1half,
)


# ============== Analysis with OnePhotonAnalyzer ==============
"""
The main purpose of OnePhotonAnalyzer class is to analyse raw output Fortran data and obtain
meaningful physical properties (cross sectionas, asymmetry parameters, Wigner delay etc).

To construct an object of OnePhotonAnalyzer class you should pass an object of OnePhoton class
with some initialized holes as input. Example below:
"""

# construct OnePhotonAnalyzer object by giving one_photon object with some initialized holes
one_photon_analyzer = OnePhotonAnalyzer(one_photon)

# 1. Photon and photoelctron kinetic energies
"""
It's usually important to check for which XUV photon/photoelectron energies our data were computed in 
the Fortran simulations. We can easily do this by calling the methods shown below:
"""
# energies for 6p_3/2 (similarly for 6p_1/2)

# XUV photon energies in eV
en = one_photon_analyzer.get_omega_eV(hole_kappa_6p3half)

# XUV photon energies in Hartree
en = one_photon_analyzer.get_omega_Hartree(hole_kappa_6p3half)

# photoelectron kinetic energies in eV
en = one_photon_analyzer.get_electron_kinetic_energy_eV(hole_kappa_6p3half)

# photoelectron kinetic energies in Hartree
en = one_photon_analyzer.get_electron_kinetic_energy_Hartree(hole_kappa_6p3half)

# 2. Integrated cross sections
"""
Usually we want to look at the integrated (over all angles) photoionisation cross sections 
after absorption of the XUV photon. Currently, our code calculates them from the probability current 
("ionisation rate"), but other options are also possbile.

There are different types of cross sections:
- Partial integrated cross section which is computed for a particular ionization channels of the 
given hole
- Total integrated cross section which is computed as a sum of all partial cross sections of the 
given hole
- Integrated photon absorption cross section which is computed as a sum of total integrated cross 
sections for all the intialized holes
- Integrated photoelectron emission cross section which we compute by interpolating holes' 
total integrated cross sections to the same photoelectron kinetic energy and summing them up
afterwards

The methods of one_photon_analyzer corresponding to the mentioned cross sections are shown below.
NOTE: All these methods also return the corresponding photoelectron kinetic energies in eV.
"""

# partial integrated cross section for the ionziation from 6p_3/2 to d_3/2
kappa_d3half = 2
en, cs = one_photon_analyzer.get_partial_integrated_cross_section(
    hole_kappa_6p3half, kappa_d3half
)

# total integrated cross section for 6p_3/2
en, cs = one_photon_analyzer.get_total_integrated_cross_section_for_hole(
    hole_kappa_6p3half
)
plt.figure("Total integrated crossection for hole 6p_3/2")
plt.plot(en, cs)
plt.title(f"Total integrated crossection for hole 6p_3/2")

# integrated photon absorption cross section
en, cs = one_photon_analyzer.get_integrated_photon_absorption_cross_section()
plt.figure("Integrated photon absorption cross section")
plt.plot(en, cs)
plt.title("Integrated photon absorption cross section")

# Integrated photonelectron emission cross section
en, cs = one_photon_analyzer.get_integrated_photoelectron_emission_cross_section()
plt.figure("Integrated photonelectron emission cross section")
plt.plot(en, cs)
plt.title(f"Integrated photonelectron emission cross section")

# 3. Angular part of a hole's cross section
"""
For in depth analysis of ionization from a hole, we should also consider angular part of its total cross
section. The angular part is computed thorugh the real asymmetry parameter. The one_photon_analyzer
objects contains method to calculate the asymmetry parameter ("_get_one_photon_asymmetry_parameter").
However, it's NOT recommended to use it directly, since it requires some preliminary preparations
(e.g. proper construction of matrix elements). Instead, you can directly use method
"get_angular_part_of_cross_section" which outputs the desired angular part. Furthermore, you can also
call "get_total_cross_section_for_hole" that gives you the total cross section including integrated and
angular part. 
NOTE: both "get_angular_part_of_cross_section" and "get_total_cross_section_for_hole"
output the corresponding photoelectron energy in eV.
"""
# specify angles to compute angular part of the hole's cross section
angles = np.array([0, 30, 45, 60])

# We also need ion charge which is usually just 1
Z = 1

# angular part of the cross section for 6p_3/2
plt.figure("Angular part of cross section for 6p_3/2")
for angle in angles:
    ekin, ang_cs = one_photon_analyzer.get_angular_part_of_cross_section(
        hole_kappa_6p3half, Z, angle
    )
    plt.plot(ekin, ang_cs, label=f"{angle}")
plt.title("Angular part of cross section for 6p_3/2")
plt.legend()

# total cross section for 6p_3/2 (angular + integrated parts)
plt.figure("Total cross section for 6p_3/2")
for angle in angles:
    ekin, ang_cs = one_photon_analyzer.get_total_cross_section_for_hole(
        hole_kappa_6p3half, Z, angle
    )
    plt.plot(ekin, ang_cs, label=f"{angle}")
plt.title("Total cross section for 6p_3/2")
plt.legend()

# 4. Wigner delay and phases
"""
The last but not least property we usually want to investigate in the one photon case is the 
Wigner delay (phase). And we actually want to consider both: integrated and angular part of the delay.
The integrated part is computed from the so-called "Wigner intensity". We can get it by calling
"_get_wigner_intensity", but it's a NOT recommended way because of the reasons similar to 
"_get_one_photon_asymmetry_parameter" method. Instead, you can call "get_integrated_wigner_delay"
to get the desired integrated part. The angular part is computed through the complex asymmetry parameter,
and you can get it by calling "get_angular_wigner_delay". one_photon_analyzer also includes method
"get_wigner_delay" that retuns the total delay: integrated + angular part.

one_photon_analyzer contains similar methods for the Wigner phase, the only difference is in the
methods' names: they contain "_phase" instead of "_delay".

The usage examples are shown below. NOTE: all the method below also return corresponding photoelectron
kinetic energies in eV. 
"""
# We also need ion charge which is usually just 1
Z = 1

# integrated Wigner delay for 6p_3/2
en, delay = one_photon_analyzer.get_integrated_wigner_delay(hole_kappa_6p3half, Z)
plt.figure("Integrated Wigner delay for 6p_3/2")
plt.plot(en, delay)
plt.title("Integrated Wigner delay for 6p_3/2")

# integrated Wigner phase for 6p_3/2
en, phase = one_photon_analyzer.get_integrated_wigner_phase(hole_kappa_6p3half, Z)
plt.figure("Integrated Wigner phase for 6p_3/2")
plt.plot(en, phase)
plt.title("Integrated Wigner phase for 6p_3/2")

# specify angles to compute angular part of the Wigner delay
angles = np.array([0, 30, 45, 60])

# angular part of Wigner delay for 6p_3/2
plt.figure("Angular part of Wigner delay for 6p_3/2")
for angle in angles:
    en, delay = one_photon_analyzer.get_angular_wigner_delay(
        hole_kappa_6p3half, Z, angle
    )
    plt.plot(en, delay, label=f"{angle}")
plt.legend()
plt.title("Angular part of Wigner delay for 6p_3/2")

# angular part of Wigner phase for 6p_3/2
plt.figure("Angular part of Wigner phase for 6p_3/2")
for angle in angles:
    en, phase = one_photon_analyzer.get_angular_wigner_phase(
        hole_kappa_6p3half, Z, angle
    )
    plt.plot(en, phase, label=f"{angle}")
plt.legend()
plt.title("Angular part of Wigner phase for 6p_3/2")

# Total (integrated + angular) Wigner delay for 6p_3/2
for angle in angles:
    en, delay = one_photon_analyzer.get_wigner_delay(hole_kappa_6p3half, Z, angle)

# Total (integrated + angular) Wigner phase for 6p_3/2
for angle in angles:
    en, phase = one_photon_analyzer.get_wigner_phase(hole_kappa_6p3half, Z, angle)

"""
Last important note about Wigner delay/phase. All the relevant methods shown above have one more
parameter "steps_per_IR_photon". It is basically the number of XUV energy steps fitted in the 
IR photon energy. You can, of course, assign a specific value to it, but if you're not sure which 
value was used during the Fotran simulations you'd better keep the default value (None). 
If the default value was given, the the program will calculate it based on the XUV energy data in the
omega.dat file and initialized value of the IR photon energy.
"""
