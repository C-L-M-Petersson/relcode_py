import numpy as np
from sympy.physics.wigner import wigner_3j
from sympy import N as sympy_to_num
from cmath import phase
from functools import partial
import sys

relcode_py_repo_path = "./relcode_py"

sys.path.append(relcode_py_repo_path)

from fortran_output_analysis.twophotons import TwoPhotons, final_kappas
from fortran_output_analysis.onephoton import final_kappas as dipole_conneted_kappas
from fortran_output_analysis.common_utility import j_from_kappa, wigner3j_numerical, ground_state_energy
from fortran_output_analysis.density_matrix import gaussian, density_matrix_pure_state, normalise_density_matrix, density_matrix_purity


# {{{ Settings:
# Fortran output directory:
data_dir = "inFiles/argon_output/"
twophoton_data_dir = data_dir+"second_photon/"

#Kinetic energies:
E_min  = 0.47
E_max  = 0.50
dE     = 0.0005
E_kins = np.arange(E_min,E_max,dE)

#The index of the IR ("abs" and 2 for absorption, "emi" and 1 for emission):
IR_label = "abs"
IR_ind   = 2

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

    #For each final (two-photon) state:
    kappas_2photon = final_kappas(kappa_g)
    for kappa_2photon in kappas_2photon:

        #For each m_j:
        m_j_max = min(j_from_kappa(kappa_g),j_from_kappa(kappa_2photon))
        for m_j in np.arange(-m_j_max,m_j_max+1):

            two_photons = TwoPhotons("Argon")
            two_photons.add_omega(data_dir + "pert_"+str(kappa_g)+"_"+str(nMinusL_g)+"/omega.dat")
            path_m_elements = twophoton_data_dir + "m_elements_eF"+str(IR_ind)+"_"+str(kappa_g)+"_"+str(nMinusL_g)+".dat"
            two_photons.add_matrix_elements(path_m_elements, IR_label, kappa_g, nMinusL_g)  # absorption

            omegas_data = two_photons.omega_Hartree
            omegas = E_kins-E_g

            ms = np.zeros(len(omegas_data))
            kappas_1photon = list(set(dipole_conneted_kappas(kappa_g)) & set(dipole_conneted_kappas(kappa_2photon)))
            for kappa_1photon in kappas_1photon:
                if(j_from_kappa(kappa_1photon)>=abs(m_j)): # Chekc that the m_j value is valid for kappa_1photon
                    fact = complex \
                            ( wigner3j_numerical( kappa_1photon, kappa_2photon, m_j )
                            * wigner3j_numerical( kappa_g      , kappa_1photon, m_j )
                            * (-1)**( j_from_kappa( kappa_2photon ) + j_from_kappa( kappa_1photon ) - 2*m_j )
                            )
                    ms_tmp = fact*(two_photons.get_matrix_element_for_intermediate_resolved_channel(kappa_g,kappa_1photon,kappa_2photon,IR_label)[0])
                    ms = ms + ms_tmp

            rho = rho + density_matrix_pure_state( E_kins, E_g, omegas_data, ms, pulse )

rho = normalise_density_matrix( rho, dE )

#Print the density matrix purity
print( density_matrix_purity( rho,dE ) )

#Print the density matrix:
for e1,rhoCol in zip(E_kins,rho):
    for e2,y in zip(E_kins,rhoCol):
        print(e1,e2,abs(y),phase(y))
    print("")
