import numpy as np
import os
import math
from scipy.special import legendre
from fortran_output_analysis.constants_and_parameters import (
    g_eV_per_Hartree,
    g_inverse_atomic_frequency_to_attoseconds,
    fine_structure,
)
from fortran_output_analysis.common_utility import (
    convert_rate_to_cross_section,
    exported_mathematica_tensor_to_python_list,
    coulomb_phase,
    delay_to_phase,
    unwrap_phase_with_nans,
    wavenumber,
    convert_amplitude_to_cross_section,
    Hole,
)
from fortran_output_analysis.onephoton.onephoton import OnePhoton

"""
This namespace contains functions for analyzing raw Fortran data in the one photon case.
"""

# TODO: probabily rewrite functions so that they take IonHole object as input


def get_omega_eV(one_photon: OnePhoton, hole: Hole):
    """
    Returns array of XUV photon energies in eV for the given hole.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters

    Returns:
    array of XUV photon energies in eV for the given hole
    """

    one_photon.assert_hole_load(hole)

    return get_omega_Hartree(one_photon, hole) * g_eV_per_Hartree


def get_omega_Hartree(one_photon: OnePhoton, hole: Hole):
    """
    Returns array of XUV photon energies in Hartree for the given hole.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters

    Returns:
    omega_Hartree - array of XUV photon energies in Hartree for the given hole
    """

    one_photon.assert_hole_load(hole)

    channel = one_photon.get_channel_for_hole(hole)

    omega_Hartree = channel.raw_data[
        :, 0
    ]  # omega energies in Hartree from the output file.

    return omega_Hartree


def get_electron_kinetic_energy_Hartree(one_photon: OnePhoton, hole: Hole):
    """
    Returns array of electron kinetic energies in Hartree for the given hole.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters

    Returns:
    array of electron kinetic energies in Hartree for the given hole
    """

    one_photon.assert_hole_load(hole)

    return get_omega_Hartree(one_photon, hole) - hole.binding_energy


def get_electron_kinetic_energy_eV(one_photon: OnePhoton, hole: Hole):
    """
    Returns array of electron kinetic energies in eV for the given hole.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters

    Returns:
    array of electron kinetic energies in eV for the given hole
    """

    one_photon.assert_hole_load(hole)

    return get_electron_kinetic_energy_Hartree(one_photon, hole) * g_eV_per_Hartree


def get_partial_integrated_cross_section_1_channel(
    one_photon: OnePhoton,
    hole: Hole,
    final_kappa,
    mode="pcur",
    divide_omega=True,
    relativistic=True,
):
    """
    Calculates integrated cross section for only one ionization channel (final state) of
    the given hole.
    Depending on conventions when creating the dipole elements in the Fortran program we
    might have to divide or multiply by the photon energy (omega) when calculating
    cross sections. Usually it is correct to divide by omega, and that is default behaviour
    of this function.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
    final_kappa - kappa value of the final state
    mode - "pcur" or "amp". "pcur" means calculation from the probability current, "amp" means
    calculcation from matrix amplitudes
    divide_omega - in "pcur" mode tells if we divide or multiply by the photon energy (omega) when
    calculating the cross section
    relativistic - in "amp" mode tells if we use relativitic wave number

    Returns:
    ekin_eV - array of electron kinetic energy
    cross_section - values of the partial integrated cross section for one channel
    """

    assert (mode == "pcur") or (
        mode == "amp"
    ), "mode parameter can only take 'pcur' or 'amp' values"

    one_photon.assert_final_kappa(hole, final_kappa)

    channel = one_photon.get_channel_for_hole(hole)

    omega = get_omega_Hartree(one_photon, hole)

    if mode == "pcur":
        rate = channel.get_rate_for_channel(final_kappa)
        cross_section = convert_rate_to_cross_section(rate, omega, divide_omega)
    else:
        ekin = omega - hole.binding_energy
        k = wavenumber(ekin, relativistic=relativistic)  # wavenumber vector
        final_state = channel.final_states[final_kappa]
        column_index = final_state.pcur_column_index
        amp_data = channel.raw_amp_data[:, column_index]
        amp_data = np.nan_to_num(
            amp_data, nan=0.0, posinf=0.0, neginf=0.0
        )  # replace all nan or inf values with 0.0
        cross_section = convert_amplitude_to_cross_section(
            amp_data, k, omega, divide_omega=divide_omega
        )

    ekin_eV = get_electron_kinetic_energy_eV(
        one_photon, hole
    )  # electron kinetic energy in eV

    return ekin_eV, cross_section


def get_partial_integrated_cross_section_multiple_channels(
    one_photon: OnePhoton,
    hole: Hole,
    final_kappas,
    mode="pcur",
    divide_omega=True,
    relativistic=True,
):
    """
    Calculates integrated cross section for several ionization channels of the given hole.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
    final_kappas - array with kappa values of the final states
    mode - "pcur" or "amp". "pcur" means calculation from the probability current, "amp" means
    calculcation from matrix amplitudes
    divide_omega - in "pcur" mode tells if we divide or multiply by the photon energy (omega) when
    calculating the cross section
    relativistic - in "amp" mode tells if we use relativitic wave number

    Returns:
    ekin_eV - array of electron kinetic energy
    cross_section - values of the partial integrated cross section for multiple channels
    """
    assert (mode == "pcur") or (
        mode == "amp"
    ), "mode parameter can only take 'pcur' or 'amp' values"

    one_photon.assert_hole_load(hole)

    ekin_eV = get_electron_kinetic_energy_eV(one_photon, hole)
    cross_section = np.zeros(len(ekin_eV))

    for final_kappa in final_kappas:
        one_photon.assert_final_kappa(hole, final_kappa)
        _, channel_cs = get_partial_integrated_cross_section_1_channel(
            one_photon,
            hole,
            final_kappa,
            mode=mode,
            divide_omega=divide_omega,
            relativistic=relativistic,
        )
        cross_section += channel_cs

    return ekin_eV, cross_section


def get_total_integrated_cross_section_for_hole(
    one_photon: OnePhoton,
    hole: Hole,
    mode="pcur",
    divide_omega=True,
    relativistic=True,
):
    """
    Calculates total integrated cross section: sums over all possible channels (final states)
    for the give hole.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
    mode - "pcur" or "amp". "pcur" means calculation from the probability current, "amp" means
    calculcation from matrix amplitudes
    divide_omega - in "pcur" mode tells if we divide or multiply by the photon energy (omega) when
    calculating the cross section
    relativistic - in "amp" mode tells if we use relativitic wave number

    Returns:
    ekin_eV - array of electron kinetic energy
    total_cs - values of the total integrated cross section
    """
    assert (mode == "pcur") or (
        mode == "amp"
    ), "mode parameter can only take 'pcur' or 'amp' values"

    one_photon.assert_hole_load(hole)

    channel = one_photon.get_channel_for_hole(hole)

    final_kappas = list(channel.final_states.keys())

    ekin_eV, total_cs = get_partial_integrated_cross_section_multiple_channels(
        one_photon,
        hole,
        final_kappas,
        mode=mode,
        divide_omega=divide_omega,
        relativistic=relativistic,
    )

    return ekin_eV, total_cs


def get_photoabsorption_cross_section(one_photon: OnePhoton, photon_energy_eV):
    """
    Computes integrated photoabsorption cross section using diagonal eigenvalues and matrix elements.

    Params:
    one_photon - object of the OnePhoton class with loaded diagonal data
    photon_energy_eV - array of photon energies for which cross section is computed in eV

    Returns:
    cross_section - array with photoabsoprtion cross section values
    """

    one_photon.assert_diag_data_load()

    M = one_photon.diag_matrix_elements
    eigvals = one_photon.diag_eigenvalues

    au_to_Mbarn = (0.529177210903) ** 2 * 100
    convert_factor = 4.0 * np.pi / 3.0 * fine_structure * au_to_Mbarn

    cross_section = np.zeros(len(photon_energy_eV))

    for i in range(len(photon_energy_eV)):
        omega_Hartree = photon_energy_eV[i] / g_eV_per_Hartree
        omega_complex = omega_Hartree + 0 * 1j
        imag_term = np.sum(M * M / (eigvals - omega_complex))
        cross_section[i] = convert_factor * np.imag(imag_term) * np.real(omega_complex)

    return cross_section


def get_integrated_photoelectron_emission_cross_section(
    one_photon: OnePhoton,
    mode_energies="ekin",
    ekin_final=None,
    mode_cs="pcur",
    divide_omega=True,
    relativistic=True,
):
    """
    Computes photoelectron emission cross section (cross section for photoelectron
    kinetic energies): sums total integrated cross sections for all loaded holes.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    mode_energies - "omega" if we want to compute cross section for the same photon energies
    (no interpolation required) or "ekin" if we want to compute cross section for
    the same photoelectron kinetic energies (interpolation required)
    ekin_final - required for "ekin" mode_energies only! Allows specifiying custom array of
    photoelectron kinetic energies to compute the cross section. If not specified,
    the function concatenates and sorts kinetic energy vectors for all holes
    mode_cs - "pcur" or "amp". "pcur" means calculation of the cross section from probability
    current, "amp" means calculcation from matrix amplitudes
    divide_omega - required for "pcur" mode_cs only! Tells if we divide or multiply by the photon
    energy (omega) when calculating the cross section
    relativistic - equired for "amp" mode_cs only! Tells if we use relativitic wave number


    Returns:
    energy - array of final energies in eV (either photon or photoelectron kinetic)
    emission_cs - values of the interpolated photonelctron emission cross section
    """

    assert (mode_energies == "omega") or (
        mode_energies == "ekin"
    ), "mode_energies parameter can only take 'omega' or 'ekin' values"

    assert (mode_cs == "pcur") or (
        mode_cs == "amp"
    ), "mode_cs parameter can only take 'pcur' or 'amp' values"

    loaded_holes = list(one_photon.channels.keys())

    assert (
        len(loaded_holes) > 0
    ), f"No holes are loaded in {one_photon.name}. Please, load at least one hole!"

    first_hole = one_photon.channels[loaded_holes[0]].hole

    if mode_energies == "ekin":
        energy_first = get_electron_kinetic_energy_eV(
            one_photon, first_hole
        )  # energy vector of the first hole
    else:
        energy_first = get_omega_eV(
            one_photon, first_hole
        )  # energy vector of the first hole

    N_energy = len(energy_first)  # length of the energy vetor
    N_holes = len(loaded_holes)  # total number of holes

    if (
        mode_energies == "ekin"
    ):  # initialize arrays for interpolation in the "ekin" mode
        holes_ekin = (
            []
        )  # list to store photoelectron kinetic energies for different holes
        holes_cs = np.zeros(
            (N_holes, N_energy)
        )  # array to store total cross sections for different holes
    else:  # initialize array to store data in the "omega" mode
        energy_eV = energy_first
        emission_cs = np.zeros(N_energy)

    for i in range(N_holes):
        hole = one_photon.channels[loaded_holes[i]].hole
        ekin, hole_cs = get_total_integrated_cross_section_for_hole(
            one_photon,
            hole,
            mode=mode_cs,
            divide_omega=divide_omega,
            relativistic=relativistic,
        )
        if mode_energies == "ekin":
            holes_ekin.append(ekin)
            holes_cs[i, :] = hole_cs
        else:
            emission_cs += hole_cs

    if mode_energies == "ekin":
        energy_eV, emission_cs = interploate_photoelectron_emission_cross_section(
            N_holes, holes_ekin, holes_cs, ekin_final
        )

    return energy_eV, emission_cs


def interploate_photoelectron_emission_cross_section(
    N_holes, holes_ekin, holes_cs, ekin_final=None
):
    """
    Peforms linear interpolation of the photoelectron emission cross sections of different
    holes to match them for the same electron kinetic energy.

    Params:
    N_holes - number of holes
    holes_ekin - array with photoelectron kinetic energies for each hole
    holes_cs - array with total integrated cross sections for each hole
    ekin_final - array with final photoelectron kinetic energies to interpolate for.
    If not specified, the function concatenates and sorts kinetic energy vectors for all
    holes

    Returns:
    ekin_final - array of final photoelectron kinetic energies
    emission_cs_interpolated - array with interpolated photonelctron emission cross section
    """

    if not ekin_final:
        ekin_concatented = np.concatenate(
            holes_ekin
        )  # concatenate all the kinetic energy arrays
        ekin_final = np.unique(
            np.sort(ekin_concatented)
        )  # sort and take the unqiue values from the concatenated array

    emission_cs_interpolated = np.zeros(ekin_final.shape)

    for i in range(N_holes):
        hole_ekin = holes_ekin[i]
        hole_cs = holes_cs[i, :]
        hole_cs_interpolated = np.interp(
            ekin_final, hole_ekin, hole_cs, left=0, right=0
        )
        emission_cs_interpolated += hole_cs_interpolated

    return ekin_final, emission_cs_interpolated


def get_matrix_elements_for_final_state(one_photon: OnePhoton, hole: Hole, final_kappa):
    """
    Computes matrix elements after one photon as amp*[e^(i*phase_of_F),
    e^(i*phase_of_G)] for the given hole and final state.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
    final_kappa - kappa value of the final state

    Returns:
    matrix elements after one photon
    """

    one_photon.assert_final_kappa(hole, final_kappa)

    channel = one_photon.get_channel_for_hole(hole)
    final_state = channel.final_states[final_kappa]
    # We assume that the data is sorted the same in amp_all and phaseF_all as in pcur_all
    # this is true at time of writing (2022-05-23).
    column_index = final_state.pcur_column_index
    return channel.raw_amp_data[:, column_index] * [
        np.exp(1j * channel.raw_phaseF_data[:, column_index]),
        np.exp(1j * channel.raw_phaseG_data[:, column_index]),
    ]


def get_matrix_elements_for_all_final_states(one_photon: OnePhoton, hole: Hole):
    """
    Computes matrix elements for all possible final states of the given hole.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters

    Returns:
    M - matrix elements
    """

    one_photon.assert_hole_load(hole)

    channel = one_photon.get_channel_for_hole(hole)
    final_kappas = channel.final_kappas(hole.kappa, only_reachable=True)

    # the first kappa from the final_kappas list
    first_of_final_kappas = final_kappas[0]

    # [0] since we are only interested in the largest relativistic component
    matrix_elements = get_matrix_elements_for_final_state(
        one_photon, hole, first_of_final_kappas
    )[0]

    M = np.zeros(
        (len(final_kappas), len(matrix_elements)), dtype="complex128"
    )  # initialize the matrix
    M[0, :] = matrix_elements  # put the matrix elements for the first kappa

    for i in range(1, len(final_kappas)):
        final_kappa = final_kappas[i]
        M[i, :] = get_matrix_elements_for_final_state(one_photon, hole, final_kappa)[0]

    return M


def get_coulomb_phase(one_photon: OnePhoton, hole: Hole, Z):
    """
    Computes Coulomb phase for all the final states of the given hole.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
    Z - charge of the ion

    Returns:
    coulomb_phase_arr - array with Coulomb phases
    """

    one_photon.assert_hole_load(hole)

    channel = one_photon.get_channel_for_hole(hole)
    final_kappas = channel.final_kappas(hole.kappa, only_reachable=True)

    ekin = get_electron_kinetic_energy_Hartree(one_photon, hole)
    coulomb_phase_arr = np.zeros(
        (len(final_kappas), len(ekin))
    )  # vector to store coulomb phase

    for i in range(len(final_kappas)):
        final_kappa = final_kappas[i]
        coulomb_phase_arr[i, :] = coulomb_phase(final_kappa, ekin, Z)

    return coulomb_phase_arr


def get_matrix_elements_with_coulomb_phase(one_photon: OnePhoton, hole: Hole, Z):
    """
    Computes matrix elements for all possible final states of the given hole
    and adds Coulomb phase to them.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
    Z - charge of the ion

    Returns:
    Matrix elements with Coulomb phase
    """

    one_photon.assert_hole_load(hole)

    M = get_matrix_elements_for_all_final_states(one_photon, hole)
    coul_phase = get_coulomb_phase(one_photon, hole, Z)  # Coulomb phase

    assert (
        M.shape == coul_phase.shape
    ), "Shapes of matrix with elements and matrix with Coulomb phase don't match!"

    return M * np.exp(1j * coul_phase)


def get_wigner_intensity(
    one_photon: OnePhoton,
    hole: Hole,
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
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
    M_emi - matrix for emission path matched to the same final photoelectron
    energies
    M_abs - matrix for absorption path matched to the same final photoelectron
    energies
    path - path to the file with coefficients for Wigner intensity calculation

    Returns:
    wigner_intensity - array with Wigner intensity values
    """

    one_photon.assert_hole_load(hole)

    assert (
        M_emi.shape == M_abs.shape
    ), "The shapes of the input matrices must be the same!"

    length = M_emi.shape[1]

    if path[-1] is not os.path.sep:
        path = path + os.path.sep

    try:
        with open(path + f"integrated_intensity_{hole.kappa}.txt", "r") as coeffs_file:
            coeffs_file_contents = coeffs_file.readlines()
    except OSError as e:
        raise NotImplementedError(
            f"The hole kappa {hole.kappa} is not yet implemented, or the file containing the coefficients could not be found!"
        )

    coeffs = exported_mathematica_tensor_to_python_list(coeffs_file_contents[2])

    wigner_intensity = np.zeros(length, dtype="complex128")
    for i in range(3):
        wigner_intensity += coeffs[i] * M_emi[i] * np.conj(M_abs[i])

    return wigner_intensity


def one_photon_asymmetry_parameter(
    one_photon: OnePhoton,
    hole: Hole,
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
    original matrix to emission and absorption paths
    (using e.g. self.__match_matrix_elements_and_energies_to_same_final_photoelectron_energy)
    and basically get two different matrices. Then pass the matrix matched for emission
    path as M1 and the one matched for absorption path as M2 and
    specify abs_emi_or_cross as "cross".

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
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

    one_photon.assert_hole_load(hole)

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
        with open(path + f"asymmetry_coeffs_2_{hole.kappa}.txt", "r") as coeffs_file:
            coeffs_file_contents = coeffs_file.readlines()
    except OSError as e:
        print(e)
        raise NotImplementedError(
            f"The hole kappa {hole.kappa} is not yet implemented, or the file containing the coefficients could not be found!"
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


def get_real_asymmetry_parameter(one_photon: OnePhoton, hole: Hole, Z):
    """
    Computes real asymmetry parameter.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
    Z - charge of the ion

    Returns:
    """

    one_photon.assert_hole_load(hole)

    ekin_eV = get_electron_kinetic_energy_eV(one_photon, hole)

    M = get_matrix_elements_with_coulomb_phase(one_photon, hole, Z)
    b2_real, _ = one_photon_asymmetry_parameter(
        one_photon, hole, M, M, "abs"
    )  # one-photon real assymetry parameter

    return ekin_eV, b2_real


def get_complex_asymmetry_parameter(
    one_photon: OnePhoton, hole: Hole, Z, steps_per_IR_photon=None
):
    """
    Computes complex asymmetry parameter.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
    Z - charge of the ion

    Returns:
    """

    one_photon.assert_hole_load(hole)

    ekin_shifted_eV, M_abs_shifted, M_emi_shifted = prepare_data_for_complex_parameter(
        one_photon, hole, Z, steps_per_IR_photon=steps_per_IR_photon
    )

    b2_complex, _ = one_photon_asymmetry_parameter(
        one_photon, hole, M_emi_shifted, M_abs_shifted, "cross"
    )

    return ekin_shifted_eV, b2_complex


def get_angular_part_of_cross_section(one_photon: OnePhoton, hole: Hole, Z, angle):
    """
    Computes angular part of the total cross section for the given hole.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
    Z - charge of the ion
    angle - angle to compute cross section

    Returns:
    ekin_eV - array of photoelectron kinetic energy in eV
    angular_part - angular part of the cross section
    """

    one_photon.assert_hole_load(hole)

    ekin_eV, b2_real = get_real_asymmetry_parameter(one_photon, hole, Z)

    angular_part = 1 + b2_real * legendre(2)(
        np.array(np.cos(math.radians(angle)))
    )  # angluar part of the cross section

    return ekin_eV, angular_part


def get_total_cross_section_for_hole(
    one_photon: OnePhoton,
    hole: Hole,
    Z,
    angle,
    mode="pcur",
    divide_omega=True,
    relativistic=True,
):
    """
    Computes total cross section (integrated part * angular part) for the given hole and
    given angle.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
    Z - charge of the ion
    angle - angle to compute cross section
    mode - for calculation of the integrated part: "pcur" or "amp".
    "pcur" means calculation from the probability current, "amp" means calculcation from matrix
    amplitudes
    divide_omega - in "pcur" mode tells if we divide or multiply by the photon energy (omega) when
    calculating the cross section
    relativistic - in "amp" mode tells if we use relativitic wave number

    Returns:
    ekin_eV - array of photoelectron kinetic energy in eV
    angular_part - total cross section
    """

    _, integrated_part = get_total_integrated_cross_section_for_hole(
        one_photon,
        hole,
        mode=mode,
        divide_omega=divide_omega,
        relativistic=relativistic,
    )
    ekin_eV, angular_part = get_angular_part_of_cross_section(
        one_photon, hole, Z, angle
    )

    return ekin_eV, integrated_part * angular_part


def get_integrated_wigner_delay(
    one_photon: OnePhoton, hole: Hole, Z, steps_per_IR_photon=None
):
    """
    Computes integrated wigner delay for the given hole.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
    Z - charge of the ion
    steps_per_IR_photon - the number of XUV energy steps fitted in the IR photon energy.
    If not specified, the the program calculates it based on the XUV energy data in the
    omega.dat file and value of the IR photon energy.

    Returns:
    ekin_shifted_eV - photoelectron kinetic energy shifted to match the same final energy
    tau_int_wigner - array with integrated Wigner delays
    """

    one_photon.assert_hole_load(hole)

    ekin_shifted_eV, M_abs_shifted, M_emi_shifted = prepare_data_for_complex_parameter(
        one_photon, hole, Z, steps_per_IR_photon
    )

    tau_int_wigner = integrated_wigner_delay_from_intensity(
        one_photon, hole, M_emi_shifted, M_abs_shifted
    )

    return ekin_shifted_eV, tau_int_wigner


def get_integrated_wigner_phase(
    one_photon: OnePhoton, hole: Hole, Z, steps_per_IR_photon=None, unwrap=True
):
    """
    Computes integrated wigner phase from the wigner delay.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
    Z - charge of the ion
    steps_per_IR_photon - the number of XUV energy steps fitted in the IR photon energy.
    If not specified, the the program calculates it based on the XUV energy data in the
    omega.dat file and value of the IR photon energy
    unwrap - if to unwrap phase using np.unwrap

    Returns:
    ekin_shifted_eV - photoelectron kinetic energy shifted to match the same final energy
    phase_int_wigner - array with integrated Wigner phases
    """

    one_photon.assert_hole_load(hole)

    g_omega_IR = one_photon.get_channel_for_hole(
        hole
    ).g_omega_IR  # frequncy of the IR photon (in Hartree)

    ekin_shifted_eV, tau_int_wigner = get_integrated_wigner_delay(
        one_photon, hole, Z, steps_per_IR_photon
    )
    phase_int_wigner = delay_to_phase(tau_int_wigner, g_omega_IR)

    if unwrap:
        phase_int_wigner = unwrap_phase_with_nans(phase_int_wigner)

    return ekin_shifted_eV, phase_int_wigner


def get_angular_wigner_delay(
    one_photon: OnePhoton, hole: Hole, Z, angle, steps_per_IR_photon=None
):
    """
    Computes angular part of the wigner delay.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
    Z - charge of the ion
    angle - angle to compute the delay
    steps_per_IR_photon - the number of XUV energy steps fitted in the IR photon energy.
    If not specified, the the program calculates it based on the XUV energy data in the
    omega.dat file and value of the IR photon energy

    Returns:
    ekin_shifted_eV - photoelectron kinetic energy shifted to match the same final energy
    tau_ang_wigner - array with angular part of Wigner delays
    """

    one_photon.assert_hole_load(hole)

    g_omega_IR = one_photon.get_channel_for_hole(
        hole
    ).g_omega_IR  # frequncy of the IR photon (in Hartree)

    ekin_shifted_eV, M_abs_shifted, M_emi_shifted = prepare_data_for_complex_parameter(
        one_photon, hole, Z, steps_per_IR_photon
    )

    tau_ang_wigner = angular_wigner_delay_from_asymmetry_parameter(
        one_photon, hole, M_emi_shifted, M_abs_shifted, angle
    )

    return ekin_shifted_eV, tau_ang_wigner


def get_angular_wigner_phase(
    one_photon: OnePhoton,
    hole: Hole,
    Z,
    angle,
    steps_per_IR_photon=None,
    unwrap=True,
):
    """
    Computes angular part of Wigner phase from Wigner delay.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
    Z - charge of the ion
    angle - angle to compute the phase
    steps_per_IR_photon - the number of XUV energy steps fitted in the IR photon energy.
    If not specified, the the program calculates it based on the XUV energy data in the
    omega.dat file and value of the IR photon energy
    unwrap - if to unwrap phase using np.unwrap

    Returns:
    ekin_shifted_eV - photoelectron kinetic energy shifted to match the same final energy
    phase_ang_wigner - array with angular part of Wigner phases
    """

    one_photon.assert_hole_load(hole)

    g_omega_IR = one_photon.get_channel_for_hole(
        hole
    ).g_omega_IR  # frequncy of the IR photon (in Hartree)

    ekin_shifted_eV, tau_ang_wigner = get_angular_wigner_delay(
        one_photon,
        hole,
        Z,
        angle,
        steps_per_IR_photon,
    )
    phase_ang_wigner = delay_to_phase(tau_ang_wigner, g_omega_IR)

    if unwrap:
        phase_ang_wigner = unwrap_phase_with_nans(phase_ang_wigner)

    return ekin_shifted_eV, phase_ang_wigner


def get_wigner_delay(
    one_photon: OnePhoton, hole: Hole, Z, angle, steps_per_IR_photon=None
):
    """
    Computes total Wigner delay: integrated + angular part.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
    Z - charge of the ion
    angle - angle to compute the delay
    steps_per_IR_photon - the number of XUV energy steps fitted in the IR photon energy.
    If not specified, the the program calculates it based on the XUV energy data in the
    omega.dat file and value of the IR photon energy

    Returns:
    ekin_shifted_eV - photoelectron kinetic energy shifted to match the same final energy
    tau_wigner - array with total Wigner delays
    """

    one_photon.assert_hole_load(hole)

    ekin_shifted_eV, M_abs_shifted, M_emi_shifted = prepare_data_for_complex_parameter(
        one_photon, hole, Z, steps_per_IR_photon
    )

    tau_int_wigner = integrated_wigner_delay_from_intensity(
        one_photon, hole, M_emi_shifted, M_abs_shifted
    )

    tau_ang_wigner = angular_wigner_delay_from_asymmetry_parameter(
        one_photon, hole, M_emi_shifted, M_abs_shifted, angle
    )

    tau_wigner = tau_int_wigner + tau_ang_wigner  # total Wigner delay

    return ekin_shifted_eV, tau_wigner


def get_wigner_phase(
    one_photon: OnePhoton,
    hole: Hole,
    Z,
    angle,
    steps_per_IR_photon=None,
    unwrap=True,
):
    """
    Computes total Wigner phase: integrated + angular part.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
    Z - charge of the ion
    angle - angle to compute the phase
    steps_per_IR_photon - the number of XUV energy steps fitted in the IR photon energy.
    If not specified, the the program calculates it based on the XUV energy data in the
    omega.dat file and value of the IR photon energy

    Returns:
    ekin_shifted_eV - photoelectron kinetic energy shifted to match the same final energy
    phase_wigner - array with total Wigner phases
    """

    one_photon.assert_hole_load(hole)

    g_omega_IR = one_photon.get_channel_for_hole(
        hole
    ).g_omega_IR  # frequncy of the IR photon (in Hartree)

    ekin_shifted_eV, tau_wigner = get_wigner_delay(
        one_photon,
        hole,
        Z,
        angle,
        steps_per_IR_photon,
    )
    phase_wigner = delay_to_phase(tau_wigner, g_omega_IR)

    if unwrap:
        phase_wigner = unwrap_phase_with_nans(phase_wigner)

    return ekin_shifted_eV, phase_wigner


def prepare_data_for_complex_parameter(
    one_photon: OnePhoton, hole: Hole, Z, steps_per_IR_photon
):
    """
    Prepares matrices and energy vector for complex asymmetry parameter computations.
    Constructs steps_per_IR_photon if not specified and matrix elements, shifts
    matrix elements and energy vector to match final photoelectron energies.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
    Z - charge of the ion
    steps_per_IR_photon - the number of XUV energy steps fitted in the IR photon energy.
    If not specified, the the program calculates it based on the XUV energy data in the
    omega.dat file and value of the IR photon energy

    Returns:
    ekin_shifted - array of photoelectron energies shifted to match the same final energies
    M_abs_shifted - matrix elements for absorption path shifted to match the same
    final energies
    M_emi_shifted - matrix elements for emission path shifted to match the same
    final energies
    """

    ekin_eV = get_electron_kinetic_energy_eV(one_photon, hole)

    g_omega_IR = one_photon.get_channel_for_hole(
        hole
    ).g_omega_IR  # frequncy of the IR photon (in Hartree)

    if not steps_per_IR_photon:
        steps_per_IR_photon = int(
            g_omega_IR / ((ekin_eV[1] - ekin_eV[0]) / g_eV_per_Hartree)
        )

    M = get_matrix_elements_with_coulomb_phase(one_photon, hole, Z)
    ekin_shifted, M_abs_shifted, M_emi_shifted = (
        match_matrix_elements_and_energies_to_same_final_photoelectron_energy(
            ekin_eV, M, M, steps_per_IR_photon
        )
    )

    return ekin_shifted, M_abs_shifted, M_emi_shifted


def integrated_wigner_delay_from_intensity(
    one_photon: OnePhoton, hole: Hole, M_emi_shifted, M_abs_shifted
):
    """
    Computes integrated Wigner delay from Wigner intenisty.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
    M_emi_shifted - matrix elements for emission path shifted to match the same
    final energies
    M_abs_shifted - matrix elements for absorption path shifted to match the same
    final energies

    Returns:
    tau_int_wigner - array with integrated Wigner delay
    """

    g_omega_IR = one_photon.get_channel_for_hole(
        hole
    ).g_omega_IR  # frequncy of the IR photon (in Hartree)

    wigner_intensity = get_wigner_intensity(
        one_photon, hole, M_emi_shifted, M_abs_shifted
    )

    tau_int_wigner = (
        g_inverse_atomic_frequency_to_attoseconds
        * np.angle(wigner_intensity)
        / (2.0 * g_omega_IR)
    )

    return tau_int_wigner


def angular_wigner_delay_from_asymmetry_parameter(
    one_photon: OnePhoton,
    hole: Hole,
    M_emi_shifted,
    M_abs_shifted,
    angle,
):
    """
    Computes angular part of Wigner delay from the complex assymetry parameter.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    hole - object of the Hole class containing hole's parameters
    M_emi_shifted - matrix elements for emission path shifted to match the same
    final energies
    M_abs_shifted - matrix elements for absorption path shifted to match the same
    final energies
    angle - angle to compute delay

    Returns:
    tau_ang_wigner - array with angular part of Wigner delay
    """

    g_omega_IR = one_photon.get_channel_for_hole(
        hole
    ).g_omega_IR  # frequncy of the IR photon (in Hartree)

    b2_complex, _ = one_photon_asymmetry_parameter(
        one_photon, hole, M_emi_shifted, M_abs_shifted, "cross"
    )  # complex assymetry parameter for one photon case

    tau_ang_wigner = (
        g_inverse_atomic_frequency_to_attoseconds
        * np.angle(
            1.0 + b2_complex * legendre(2)(np.array(np.cos(math.radians(angle))))
        )
        / (2 * g_omega_IR)
    )

    return tau_ang_wigner


def match_matrix_elements_and_energies_to_same_final_photoelectron_energy(
    energies, M_abs, M_emi, steps_per_IR_photon
):
    """
    Shifts matrix element and energy array so that the same index in all
    of them corresponds to the same final photoelectron energy

    Params:
    energies - array of energies
    M_abs - unshifted matrix elements for abosrption path
    M_emi - unshifted matrix elements for emission path
    steps_per_IR_photon - the number of XUV energy steps fitted in the IR photon energy.
    If not specified, the the program calculates it based on the XUV energy data in the
    omega.dat file and value of the IR photon energy

    Returns:
    energies_shifted - array of photoelectron energies shifted to match the same final
    energies
    M_abs_shifted - matrix elements for absorption path shifted to match the same
    final energies
    M_emi_shifted - matrix elements for emission path shifted to match the same
    final energies
    """

    energies_shifted = energies[
        steps_per_IR_photon : (len(energies) - steps_per_IR_photon)
    ]

    M_abs_shifted = np.zeros(
        (M_abs.shape[0], M_abs.shape[1] - 2 * steps_per_IR_photon),
        dtype="complex128",
    )
    for i in range(M_abs.shape[0]):
        M_abs_shifted[i, :] = M_abs[i, : (M_abs.shape[1] - 2 * steps_per_IR_photon)]

    M_emi_shifted = np.zeros(
        (M_emi.shape[0], M_emi.shape[1] - 2 * steps_per_IR_photon),
        dtype="complex128",
    )
    for i in range(M_emi.shape[0]):
        M_emi_shifted[i, :] = M_emi[i, 2 * steps_per_IR_photon :]

    return energies_shifted, M_abs_shifted, M_emi_shifted
