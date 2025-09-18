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

    # rho = np.outer(interpolated_mat_elems, np.conj(interpolated_mat_elems))
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

# ==================================================================================================
#
# ==================================================================================================

def SchmidtDecomp( rho, lambda_lim = None ):
    """Returns the Schmidt coefficients S and states U = V^\dagger of the hermitian density matrix rho.
    Arguments:
        - rho : [[complex float]]
            The density matrix.

        - lambda_lim : float (optional)
            Tolerance values for the singular values lambda, i.e. it filters out states with Schmidt coefficients less than lambda_lim. By default it returns all states.
    
    """

    U, S, V = np.linalg.svd(rho, hermitian = True)

    if lambda_lim != None:
        for i in range(len(S)):
            if S[i] < lambda_lim:
                U = U[:, :i]
                S = S[:i]
                break

    return S, U

def RhoFromSchmidtDecomp( S, U, idxs = None ):
    """Returns the density matrix, rho, using the Schmidt coefficients S and states U.
    Arguments:
        - S : [float]
            Vector of Schmidt coefficients.
    
        - U : [[complex float]]
            Matrix containing reduced Schmidt states.

        - idxs : [int] (optional)
            For reconstructing the density matrix using certain indices instead of the full U and S.
    """

    if idxs != None:

        rho = np.zeros((np.shape(U)[1], np.shape(U)[1]), dtype = np.complex128)
        for i in idxs:
            rho += S[i] * np.outer(U[:, i], np.conj(U[:, i]))

    else:
        rho = U*S@np.conj(U.T)

    return rho

# ==================================================================================================
#
# ==================================================================================================

def SchmidtDFT(State, t, omega, inverse = False):
    """Returns the Discrete Fourier Transform for the input Schmidt state.
    Arguments:
        - State : [complex float]
            Input Schmidt state.

        - t : [float]
            Time.

        - omega : [float]
            Frequency.

        - inverse : bool (optional)
            Choose between DFT and IDFT.
    """

    t = np.asarray(t)
    omega = np.asarray(omega)

    if inverse:
        
        FT = np.zeros(len(t), dtype = complex)

        for t_idx, t_value in enumerate(t):
            integrand = State * np.exp(1j * omega * t_value)
            FT[t_idx] = np.trapz(integrand, x = omega)

        return FT / (2*np.pi)

    else:

        FT = np.zeros(len(omega), dtype = complex)

        for omega_idx, omega_value in enumerate(omega):
            integrand = State * np.exp(-1j * omega_value * t)
            FT[omega_idx] = np.trapz(integrand, x = t)

        return FT
    
def FullSchmidtDFT(U, t, omega, inverse = False):
    """Returns the Discrete Fourier Transform for all Schmidt states in U as coloumn vectors.
    Arguments:
        - U : [[complex float]]
            Matrix of Schmidt states. Note: The Schmidt states must be the coloumn vectors of U.

        - t : [float]
            Time.

        - omega : [float]
            Frequency.

        - inverse : bool (optional)
            Choose between DFT and IDFT.
    """

    if inverse:

        colU = np.shape(U)[1]
        U_IDFT = np.zeros((len(t), colU), dtype = complex)

        for i in range(colU):
            State = U[:, i]

            U_IDFT[:, i] = SchmidtDFT(State, t, omega, inverse = True)

        return U_IDFT

    else:

        colU = np.shape(U)[1]
        U_DFT = np.zeros((len(t), colU), dtype = complex)

        for i in range(colU):
            State = U[:, i]

            U_DFT[:, i] = SchmidtDFT(State, t, omega)

        return U_DFT
