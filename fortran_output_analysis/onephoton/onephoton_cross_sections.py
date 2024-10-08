import numpy as np
import os
import math
from scipy.special import legendre
from fortran_output_analysis.constants_and_parameters import (
    g_eV_per_Hartree,
    fine_structure,
)
from fortran_output_analysis.common_utility import (
    convert_rate_to_cross_section,
    wavenumber,
    convert_amplitude_to_cross_section,
    Hole,
)
from fortran_output_analysis.onephoton.onephoton import OnePhoton
from fortran_output_analysis.onephoton.onephoton_utilities import (
    get_omega_Hartree,
    get_electron_kinetic_energy_eV,
    get_omega_eV,
    get_electron_kinetic_energy_Hartree,
)
from fortran_output_analysis.onephoton.onephoton_asymmetry_parameters import (
    get_real_asymmetry_parameter,
)

"""
This namespace contains functions for analyzing cross sections based on the data from
OnePhoton object.
"""


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
        ekin = get_electron_kinetic_energy_Hartree(one_photon, hole)
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
    photon_energy_eV - array of photon energies for which the cross section is computed in eV.

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
        ekin_final = np.sort(
            np.unique(ekin_concatented)
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
