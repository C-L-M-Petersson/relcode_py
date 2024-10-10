import numpy as np
import os
from typing import Optional
from fortran_output_analysis.common_utility import (
    exported_mathematica_tensor_to_python_list,
    Hole,
)
from fortran_output_analysis.onephoton.onephoton import OnePhoton
from fortran_output_analysis.onephoton.onephoton_utilities import (
    get_electron_kinetic_energy_eV,
    get_matrix_elements_with_coulomb_phase,
    get_prepared_matrices,
)

"""
This namespace contains functions for analyzing asymmetry parameters based on the data from 
the OnePhoton object.
"""


def one_photon_asymmetry_parameter(
    hole_kappa,
    M1,
    M2,
    abs_emi_or_cross,
    path=os.path.join(
        os.path.sep.join(
            os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1]
        ),
        "formula_coefficients",
        "one_photon",
        "asymmetry_coeffs",
    ),
    threshold=1e-10,
):
    """
    Computes the value of the asymmetry parameter for a state defined by hole_kappa in
    the one photon case.

    M1 and M2 contains the matrix elements and other phases of the wave function organized
    according to their final kappa like so:
    m = |hole_kappa|
    s = sign(hole_kappa)
    M = [s(m-1), -sm, s(m+1)]

    If you want to calculate real asymmetry parameter, then simply pass the same matrix to
    M1 and M2 and specify abs_emi_or_cross as "abs" or "emi".
    If you want to calculate complex asymmetry parameter, then you need to match the
    original matrix to emission and absorption paths and basically get two different matrices.
    Then pass the matrix matched for emission path as M1 and the one matched for absorption path
    as M2 and specify abs_emi_or_cross as "cross".

    Params:
    hole_kappa - kappa value of the hole
    M1, M2 - either the same one photon matrix or two matrices matched for emission and
    absoprtion paths
    abs_emi_or_cross - specify "abs" or "emi" when provide the same matrix and "cross"
    otherwise
    path - path to the file with coefficients for asymmetry parameter calculation
    threshold - required to check if the imaginary part of the real asymmetry parameter
    is small (less than this threshold value), since in theory it should be 0

    Returns:
    parameter - array with asymmetry parameter values
    label - a string specifying which asymmetry parameter (real or complex) was computed
    """

    if (
        abs_emi_or_cross != "abs"
        and abs_emi_or_cross != "emi"
        and abs_emi_or_cross != "cross"
    ):
        raise ValueError(
            f"abs_emi_or_cross can only be 'abs', 'emi', or 'cross' not {abs_emi_or_cross}"
        )

    if path[-1] is not os.path.sep:
        path = path + os.path.sep

    assert M1.shape == M2.shape, "The shapes of the input matrices must be the same!"

    data_size = M1.shape[1]

    # Try opening the needed file.
    try:
        with open(path + f"asymmetry_coeffs_2_{hole_kappa}.txt", "r") as coeffs_file:
            coeffs_file_contents = coeffs_file.readlines()
    except OSError as e:
        print(e)
        raise NotImplementedError(
            f"The hole kappa {hole_kappa} is not yet implemented, or the file containing the coefficients could not be found!"
        )

    # Read in the coefficients in front of all the different combinations of matrix elements in the numerator.
    numerator_coeffs = np.array(
        exported_mathematica_tensor_to_python_list(coeffs_file_contents[3])
    )

    # Read in the coefficients in front of the absolute values in the denominator.
    denominator_coeffs = exported_mathematica_tensor_to_python_list(
        coeffs_file_contents[4]
    )

    numerator = np.zeros(data_size, dtype="complex128")
    denominator = np.zeros(data_size, dtype="complex128")
    for i in range(3):
        denominator += denominator_coeffs[i] * M1[i] * np.conj(M2[i])
        for j in range(3):
            numerator += numerator_coeffs[i, j] * M1[i] * np.conj(M2[j])

    parameter = numerator / denominator

    if abs_emi_or_cross != "cross":
        # When looking at the asymmetry parameter from the diagonal part
        # or the full cross part, the result is a real number
        values = parameter[
            ~np.isnan(parameter)
        ]  # Filter out the nans first, as they mess up boolean expressions (nan is not itself).
        assert all(
            np.abs(np.imag(values)) < threshold
        ), "The asymmetry parameter had a non-zero imaginary part when it shouldn't. Check the input matrix elements or change the threshold for the allowed size of the imaginary part"
        parameter = np.real(parameter)

    if abs_emi_or_cross == "cross":
        abs_emi_or_cross = "complex"

    label = f"$\\beta_2^{{{abs_emi_or_cross}}}$"

    return parameter, label


def get_real_asymmetry_parameter(one_photon: OnePhoton, n_qn, hole_kappa, Z):
    """
    Computes real asymmetry parameter.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion

    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    b2_real - values of the real asymmetry parameter
    """

    one_photon.assert_hole_load(n_qn, hole_kappa)

    ekin_eV = get_electron_kinetic_energy_eV(one_photon, n_qn, hole_kappa)

    M = get_matrix_elements_with_coulomb_phase(one_photon, n_qn, hole_kappa, Z)
    b2_real, _ = one_photon_asymmetry_parameter(
        hole_kappa, M, M, "abs"
    )  # one-photon real assymetry parameter

    return ekin_eV, b2_real


def get_complex_asymmetry_parameter(
    one_photon_1: OnePhoton,
    n_qn,
    hole_kappa,
    Z,
    one_photon_2: Optional[OnePhoton] = None,
    steps_per_IR_photon=None,
    energies_mode="both",
):
    """
    Computes complex asymmetry parameter. Can compute for 1 or 2 simulations.
    If 2 simulations are provided, then the first one_photon object corresponds
    to emission path, while the second one_photon object corresponds to absorption path.

    Params:
    one_photon_1 - object of the OnePhoton class corresponding to the first simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    one_photon_2 - object of the OnePhoton class specified if we want to consider 2 simulations
    (first for emission, second for absorption)
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.


    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    b2_complex - values of the complex asymmetry parameter
    """

    ekin_eV, M_emi_matched, M_abs_matched = get_prepared_matrices(
        one_photon_1,
        n_qn,
        hole_kappa,
        Z,
        one_photon_2=one_photon_2,
        steps_per_IR_photon=steps_per_IR_photon,
        energies_mode=energies_mode,
    )

    b2_complex, _ = one_photon_asymmetry_parameter(
        hole_kappa, M_emi_matched, M_abs_matched, "cross"
    )

    return ekin_eV, b2_complex
