import numpy as np
from cmath import phase
from functools import partial
import sys

relcode_py_repo_path = "./relcode_py"

sys.path.append(relcode_py_repo_path)

from fortran_output_analysis.common_utility import kappa_from_l_and_j, ground_state_energy
from fortran_output_analysis.density_matrix import gaussian, density_matrix_pure_state, normalise_density_matrix, density_matrix_purity
from fortran_output_analysis.onephoton import OnePhoton, final_kappas


# {{{ Settings:
# Fortran output directory:
data_dir = "inFiles/argon_output/"

#Kinetic energies:
E_min  = 0.47
E_max  = 0.50
dE     = 0.0005
E_kins = np.arange(E_min,E_max,dE)

#The pulse:
FWHM     = 0.005
E0       = 29.25/27.211
pulse = partial(gaussian,FWHM,E0)
# }}}


#The density matrix:
rho = np.zeros((len(E_kins),len(E_kins)),dtype=complex)

#For each parent ion
kappas_g   = [-2,1]
nMinusLs_g = [ 2,2]
for kappa_g,nMinusL_g in zip(kappas_g,nMinusLs_g):

    E_g = ground_state_energy(data_dir,kappa_g,nMinusL_g)

    #For each final (one-photon) state:
    kappas_1photon = final_kappas(kappa_g)
    for kappa_1photon in kappas_1photon:

        one_photon = OnePhoton("Argon")
        one_photon.add_channels_for_hole(data_dir+"pert_"+str(kappa_g)+"_"+str(nMinusL_g)+"/pcur_all.dat", kappa_g, nMinusL_g)

        omegas_data = one_photon.get_omega_Hartree()
        omegas = E_kins-E_g

        #NOTE: THE CHANNEL PHASE IS NOT ADDED!
        ms = one_photon.get_matrix_element_with_phase_for_channel( kappa_g,kappa_1photon )[0]

        rho = rho + density_matrix_pure_state( E_kins, E_g, omegas_data, ms, pulse )

rho = normalise_density_matrix( rho, dE )

#Print the density matrix purity:
print( density_matrix_purity( rho,dE ) )

#Print the density matrix:
for e1,rhoCol in zip(E_kins,rho):
    for e2,y in zip(E_kins,rhoCol):
        print(e1,e2,abs(y),phase(y))
    print("")
