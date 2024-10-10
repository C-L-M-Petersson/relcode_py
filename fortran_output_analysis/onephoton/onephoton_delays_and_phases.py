import numpy as np
import os
import math
from typing import Optional
from scipy.special import legendre
from fortran_output_analysis.constants_and_parameters import (
    g_inverse_atomic_frequency_to_attoseconds,
)
from fortran_output_analysis.common_utility import (
    delay_to_phase,
    unwrap_phase_with_nans,
    exported_mathematica_tensor_to_python_list,
    compute_omega_diff,
)
from fortran_output_analysis.onephoton.onephoton import OnePhoton
from fortran_output_analysis.onephoton.onephoton_utilities import get_prepared_matrices
from fortran_output_analysis.onephoton.onephoton_asymmetry_parameters import (
    one_photon_asymmetry_parameter,
)

"""
This namespace contains functions for analyzing delays and phases based on the data from 
the OnePhoton object.
"""


def get_wigner_intensity(
    hole_kappa,
    M_emi,
    M_abs,
    path=os.path.join(
        os.path.sep.join(
            os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1]
        ),
        "formula_coefficients",
        "one_photon",
        "integrated_intensity",
    ),
):
    """
    Computes Wigner intensity for a photoelectron that has absorbed one photon.

    Params:
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    M_emi - matrix for emission path matched to the same final photoelectron
    energies
    M_abs - matrix for absorption path matched to the same final photoelectron
    energies
    path - path to the file with coefficients for Wigner intensity calculation

    Returns:
    wigner_intensity - array with Wigner intensity values
    """

    assert (
        M_emi.shape == M_abs.shape
    ), "The shapes of the input matrices must be the same!"

    length = M_emi.shape[1]

    if path[-1] is not os.path.sep:
        path = path + os.path.sep

    try:
        with open(path + f"integrated_intensity_{hole_kappa}.txt", "r") as coeffs_file:
            coeffs_file_contents = coeffs_file.readlines()
    except OSError as e:
        raise NotImplementedError(
            f"The hole kappa {hole_kappa} is not yet implemented, or the file containing the coefficients could not be found!"
        )

    coeffs = exported_mathematica_tensor_to_python_list(coeffs_file_contents[2])

    wigner_intensity = np.zeros(length, dtype="complex128")
    for i in range(3):
        wigner_intensity += coeffs[i] * M_emi[i] * np.conj(M_abs[i])

    return wigner_intensity


def get_integrated_wigner_delay(
    one_photon_1: OnePhoton,
    n_qn,
    hole_kappa,
    Z,
    one_photon_2: Optional[OnePhoton] = None,
    steps_per_IR_photon=None,
    energies_mode="both",
):
    """
    Computes integrated Wigner delay. Can compute for 1 or 2 simulations.
    If 2 simulations are provided, then the first one_photon object corresponds
    to emission path, while the second one_photon object corresponds to absorption path.

    Params:
    one_photon_1 - object of the OnePhoton class corresponding to one simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    one_photon_2 - second object of the OnePhoton class if we want to consider 2 simulations
    (first for emission, second for absorption)
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.

    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    tau_int_wigner - values of the integrated Wigner delay
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

    omega_diff = compute_omega_diff(one_photon_1, photon_object_2=one_photon_2)

    tau_int_wigner = integrated_wigner_delay_from_intensity(
        hole_kappa, omega_diff, M_emi_matched, M_abs_matched
    )

    return ekin_eV, tau_int_wigner


def get_integrated_wigner_phase(
    one_photon_1: OnePhoton,
    n_qn,
    hole_kappa,
    Z,
    one_photon_2: Optional[OnePhoton] = None,
    steps_per_IR_photon=None,
    energies_mode="both",
    unwrap=True,
):
    """
    Computes integrated wigner phase from integrated wigner delay. Can compute for 1 or 2
    simulations. If 2 simulations are provided, then the first one_photon object corresponds
    to emission path, while the second one_photon object corresponds to absorption path.

    Params:
    one_photon_1 - object of the OnePhoton class corresponding to one simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    one_photon_2 - second object of the OnePhoton class if we want to consider 2 simulations
    (first for emission, second for absorption)
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.
    unwrap - if to unwrap phase using np.unwrap

    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    phase_int_wigner - values of the integrated Wigner phase
    """

    ekin_eV, tau_int_wigner = get_integrated_wigner_delay(
        one_photon_1,
        n_qn,
        hole_kappa,
        Z,
        one_photon_2=one_photon_2,
        steps_per_IR_photon=steps_per_IR_photon,
        energies_mode=energies_mode,
    )

    omega_diff = compute_omega_diff(one_photon_1, photon_object_2=one_photon_2)

    phase_int_wigner = delay_to_phase(tau_int_wigner, omega_diff)

    if unwrap:
        phase_int_wigner = unwrap_phase_with_nans(phase_int_wigner)

    return ekin_eV, phase_int_wigner


def get_angular_wigner_delay(
    one_photon_1: OnePhoton,
    n_qn,
    hole_kappa,
    Z,
    angle,
    one_photon_2: Optional[OnePhoton] = None,
    steps_per_IR_photon=None,
    energies_mode="both",
):
    """
    Computes angular part of Wigner delay. Can compute for 1 or 2 simulations.
    If 2 simulations are provided, then the first one_photon object corresponds
    to emission path, while the second one_photon object corresponds to absorption path.

    Params:
    one_photon_1 - object of the OnePhoton class corresponding to one simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    angle - angle to compute the delay
    one_photon_2 - second object of the OnePhoton class if we want to consider 2 simulations
    (first for emission, second for absorption)
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.

    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    tau_ang_wigner - values of the angular part of Wigner delay
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

    omega_diff = compute_omega_diff(one_photon_1, photon_object_2=one_photon_2)

    tau_ang_wigner = angular_wigner_delay_from_asymmetry_parameter(
        hole_kappa, omega_diff, M_emi_matched, M_abs_matched, angle
    )

    return ekin_eV, tau_ang_wigner


def get_angular_wigner_phase(
    one_photon_1: OnePhoton,
    n_qn,
    hole_kappa,
    Z,
    angle,
    one_photon_2: Optional[OnePhoton] = None,
    steps_per_IR_photon=None,
    energies_mode="both",
    unwrap=True,
):
    """
    Computes angluar part of Wigner phase from the angular part of Wigner delay.
    Can compute for 1 or 2 simulations. If 2 simulations are provided, then the first one_photon
    object corresponds to emission path, while the second one_photon object corresponds to
    absorption path.

    Params:
    one_photon_1 - object of the OnePhoton class corresponding to one simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    angle - angle to compute phase
    one_photon_2 - second object of the OnePhoton class if we want to consider 2 simulations
    (first for emission, second for absorption)
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.
    unwrap - if to unwrap phase using np.unwrap

    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    phase_ang_wigner - values of the angular part of Wigner phase
    """

    ekin_eV, tau_ang_wigner = get_angular_wigner_delay(
        one_photon_1,
        n_qn,
        hole_kappa,
        Z,
        angle,
        one_photon_2=one_photon_2,
        steps_per_IR_photon=steps_per_IR_photon,
        energies_mode=energies_mode,
    )

    omega_diff = compute_omega_diff(one_photon_1, photon_object_2=one_photon_2)

    phase_ang_wigner = delay_to_phase(tau_ang_wigner, omega_diff)

    if unwrap:
        phase_ang_wigner = unwrap_phase_with_nans(phase_ang_wigner)

    return ekin_eV, phase_ang_wigner


def get_wigner_delay(
    one_photon_1: OnePhoton,
    n_qn,
    hole_kappa,
    Z,
    angle,
    one_photon_2: Optional[OnePhoton] = None,
    steps_per_IR_photon=None,
    energies_mode="both",
):
    """
    Computes total Wigner delay: integrated + angular part. Can compute for 1 or 2 simulations.
    If 2 simulations are provided, then the first one_photon object corresponds to emission path,
    while the second one_photon object corresponds to absorption path.

    Params:
    one_photon_1 - object of the OnePhoton class corresponding to one simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    angle - angle to compute the delay
    one_photon_2 - second object of the OnePhoton class if we want to consider 2 simulations
    (first for emission, second for absorption)
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.

    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    tau_wigner - array with total Wigner delays
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

    omega_diff = compute_omega_diff(one_photon_1, photon_object_2=one_photon_2)

    tau_int_wigner = integrated_wigner_delay_from_intensity(
        hole_kappa, omega_diff, M_emi_matched, M_abs_matched
    )

    tau_ang_wigner = angular_wigner_delay_from_asymmetry_parameter(
        hole_kappa, omega_diff, M_emi_matched, M_abs_matched, angle
    )

    tau_wigner = tau_int_wigner + tau_ang_wigner  # total Wigner delay

    return ekin_eV, tau_wigner


def get_wigner_phase(
    one_photon_1: OnePhoton,
    n_qn,
    hole_kappa,
    Z,
    angle,
    one_photon_2: Optional[OnePhoton] = None,
    steps_per_IR_photon=None,
    energies_mode="both",
    unwrap=True,
):
    """
    Computes total Wigner phase: integrated + angular part from total Wigner delay.

    Params:
    one_photon_1 - object of the OnePhoton class corresponding to one simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    angle - angle to compute the delay
    one_photon_2 - second object of the OnePhoton class if we want to consider 2 simulations
    (first for emission, second for absorption)
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.
    unwrap - if to unwrap phase using np.unwrap

    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    phase_wigner - array with total Wigner phases
    """

    ekin_eV, tau_wigner = get_wigner_delay(
        one_photon_1,
        n_qn,
        hole_kappa,
        Z,
        angle,
        one_photon_2=one_photon_2,
        steps_per_IR_photon=steps_per_IR_photon,
        energies_mode=energies_mode,
    )

    omega_diff = compute_omega_diff(one_photon_1, photon_object_2=one_photon_2)

    phase_wigner = delay_to_phase(tau_wigner, omega_diff)

    if unwrap:
        phase_wigner = unwrap_phase_with_nans(phase_wigner)

    return ekin_eV, phase_wigner


def integrated_wigner_delay_from_intensity(
    hole_kappa, omega_diff, M_emi_matched, M_abs_matched
):
    """
    Computes integrated Wigner delay from Wigner intenisty.

    Params:
    hole_kappa - kappa value of the hole
    omega_diff - energy difference between absorption and emission paths
    M_emi_matched - matrix elements for emission path matched to the final energies
    M_abs_matched - matrix elements for absorption path matched to the final energies

    Returns:
    tau_int_wigner - array with integrated Wigner delay
    """

    wigner_intensity = get_wigner_intensity(hole_kappa, M_emi_matched, M_abs_matched)

    tau_int_wigner = (
        g_inverse_atomic_frequency_to_attoseconds
        * np.angle(wigner_intensity)
        / omega_diff
    )

    return tau_int_wigner


def angular_wigner_delay_from_asymmetry_parameter(
    hole_kappa,
    omega_diff,
    M_emi_matched,
    M_abs_matched,
    angle,
):
    """
    Computes angular part of Wigner delay from the complex assymetry parameter.

    Params:
    hole_kappa - kappa value of the hole
    omega_diff - energy difference between absorption and emission paths
    M_emi_matched - matrix elements for emission path matched to the final energies
    M_abs_matched - matrix elements for absorption path matched to the final energies
    angle - angle to compute delay

    Returns:
    tau_ang_wigner - array with angular part of Wigner delay
    """

    b2_complex, _ = one_photon_asymmetry_parameter(
        hole_kappa, M_emi_matched, M_abs_matched, "cross"
    )  # complex assymetry parameter for one photon case

    tau_ang_wigner = (
        g_inverse_atomic_frequency_to_attoseconds
        * np.angle(
            1.0 + b2_complex * legendre(2)(np.array(np.cos(math.radians(angle))))
        )
        / omega_diff
    )

    return tau_ang_wigner
