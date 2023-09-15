import numpy as np


def gaussian( FWHM, E0, E ):
    """A monochromatic gaussian pulse
    Arguments:
        - FWHM : Double
            The spectral FWHM

        - E0   : Double
            The central energy

        - E    : Double
            The energy
    """
    return np.exp( -2*np.log(2)*(E-E0)**2/FWHM**2 )


# ==================================================================================================
#
# ==================================================================================================
def density_matrix_pure_state( E_kins
                             , E_g
                             , omegas_mat_elems
                             , mat_elems
                             , pulse = lambda x : 1
                             ):
    """Creates the density matrix for a pure state
    Arguments:
        - E_kins           : [float]
            The kinetic energy basis. The density matrix is linearly
            interpolated on this basis.

        - E_g              : float
            The ground state energy, as given in hf_energies_kappa_$KAPPA.dat

        - omegas_mat_elems : [float]
            The state one-photon photon energies, as given in omega.dat

        - mat_elems        : [complex float]
            The matrix elements, corresponding to the energies in
            omegas_mat_elems.

        - pulse            : float -> float
            The spectral pulse shape. If no value is given, the pulse is asumed
            to be one everywhere
    """
    interpolated_mat_elems = np.interp( E_kins-E_g, omegas_mat_elems, mat_elems, left=0, right=0 ) * pulse( E_kins-E_g )

    def density_matrix_col( interpolated_mat_elem ):
        return interpolated_mat_elems * np.conj( interpolated_mat_elem )

    rho = np.zeros((len(E_kins),len(E_kins)),dtype=complex)
    for i in range(len(interpolated_mat_elems)):
        rho[i] = density_matrix_col( interpolated_mat_elems[i] )

    return rho


# ==================================================================================================
#
# ==================================================================================================
def density_matrix_trace( rho, dE = 1 ):
    """Returns the trace of a density matrix
    Arguments:
        - rho : [[comlex float]]
            The density matrix.

        - dE  : float
            The energy step size. Default value 1
    """
    trace = 0
    for i in range(0,len(rho)):
        trace = trace + abs(rho[i][i])*dE

    return trace

def normalise_density_matrix( rho, dE = 1 ):
    """Returns the normalised density matrix
    Arguments:
        - rho : [[comlex float]]
            The density matrix.

        - dE  : float
            The energy step size. Default value 1
    """
    return rho/density_matrix_trace( rho,dE )

def density_matrix_purity( rho, dE = 1 ):
    """Returns the (assumed normalised) density matrix purity.
    Arguments:
        - rho : [[comlex float]]
            The density matrix.

        - dE  : float
            The energy step size. Default value 1
    """
    return density_matrix_trace( np.matmul( rho, rho ), dE )*dE
