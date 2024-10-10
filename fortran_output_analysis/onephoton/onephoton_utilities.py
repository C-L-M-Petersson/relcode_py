import numpy as np
from typing import Optional
from fortran_output_analysis.constants_and_parameters import g_eV_per_Hartree

from fortran_output_analysis.common_utility import (
    coulomb_phase,
    final_energies_for_matching_1sim,
    match_matrix_elements_1sim,
    final_energies_for_matching_2sim,
    match_matrix_elements_2sim,
)
from fortran_output_analysis.onephoton.onephoton import OnePhoton

"""
This name space contains functions that may be required across different sections of one photon 
analysis (e.g. in analysis of cross sections, asymmetry parameters, delays).
"""


def get_omega_eV(one_photon: OnePhoton, n_qn, hole_kappa):
    """
    Returns array of XUV photon energies in eV for the given hole.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole

    Returns:
    array of XUV photon energies in eV for the given hole
    """

    one_photon.assert_hole_load(n_qn, hole_kappa)

    return get_omega_Hartree(one_photon, n_qn, hole_kappa) * g_eV_per_Hartree


def get_omega_Hartree(one_photon: OnePhoton, n_qn, hole_kappa):
    """
    Returns array of XUV photon energies in Hartree for the given hole.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole

    Returns:
    omega_Hartree - array of XUV photon energies in Hartree for the given hole
    """

    one_photon.assert_hole_load(n_qn, hole_kappa)

    channels = one_photon.get_channels_for_hole(n_qn, hole_kappa)

    omega_Hartree = channels.raw_data[
        :, 0
    ]  # omega energies in Hartree from the output file.

    return omega_Hartree


def get_electron_kinetic_energy_Hartree(one_photon: OnePhoton, n_qn, hole_kappa):
    """
    Returns array of electron kinetic energies in Hartree for the given hole.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole

    Returns:
    array of electron kinetic energies in Hartree for the given hole
    """

    one_photon.assert_hole_load(n_qn, hole_kappa)

    hole = one_photon.get_hole_object(n_qn, hole_kappa)

    if not hole.binding_energy:
        raise RuntimeError(f"The binding energy for {hole.name} is not initialized!")

    return get_omega_Hartree(one_photon, n_qn, hole_kappa) - hole.binding_energy


def get_electron_kinetic_energy_eV(one_photon: OnePhoton, n_qn, hole_kappa):
    """
    Returns array of electron kinetic energies in eV for the given hole.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole

    Returns:
    array of electron kinetic energies in eV for the given hole
    """

    one_photon.assert_hole_load(n_qn, hole_kappa)

    return (
        get_electron_kinetic_energy_Hartree(one_photon, n_qn, hole_kappa)
        * g_eV_per_Hartree
    )


def get_matrix_elements_for_final_state(
    one_photon: OnePhoton, n_qn, hole_kappa, final_kappa
):
    """
    Computes matrix elements after one photon as amp*[e^(i*phase_of_F),
    e^(i*phase_of_G)] for the given hole and final state.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    final_kappa - kappa value of the final state

    Returns:
    matrix elements after one photon
    """

    one_photon.assert_final_kappa(n_qn, hole_kappa, final_kappa)

    channels = one_photon.get_channels_for_hole(n_qn, hole_kappa)
    final_state = channels.final_states[final_kappa]
    # We assume that the data is sorted the same in amp_all and phaseF_all as in pcur_all
    # this is true at time of writing (2022-05-23).
    column_index = final_state.pcur_column_index
    return channels.raw_amp_data[:, column_index] * [
        np.exp(1j * channels.raw_phaseF_data[:, column_index]),
        np.exp(1j * channels.raw_phaseG_data[:, column_index]),
    ]


def get_matrix_elements_for_all_final_states(one_photon: OnePhoton, n_qn, hole_kappa):
    """
    Computes matrix elements for all possible final states of the given hole.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole

    Returns:
    M - matrix elements
    """

    one_photon.assert_hole_load(n_qn, hole_kappa)

    channels = one_photon.get_channels_for_hole(n_qn, hole_kappa)
    final_kappas = channels.final_kappas(hole_kappa, only_reachable=True)

    # the first kappa from the final_kappas list
    first_of_final_kappas = final_kappas[0]

    # [0] since we are only interested in the largest relativistic component
    matrix_elements = get_matrix_elements_for_final_state(
        one_photon, n_qn, hole_kappa, first_of_final_kappas
    )[0]

    M = np.zeros(
        (len(final_kappas), len(matrix_elements)), dtype="complex128"
    )  # initialize the matrix
    M[0, :] = matrix_elements  # put the matrix elements for the first kappa

    for i in range(1, len(final_kappas)):
        final_kappa = final_kappas[i]
        M[i, :] = get_matrix_elements_for_final_state(
            one_photon, n_qn, hole_kappa, final_kappa
        )[0]

    return M


def get_coulomb_phase(one_photon: OnePhoton, n_qn, hole_kappa, Z):
    """
    Computes Coulomb phase for all the final states of the given hole.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion

    Returns:
    coulomb_phase_arr - array with Coulomb phases
    """

    one_photon.assert_hole_load(n_qn, hole_kappa)

    channels = one_photon.get_channels_for_hole(n_qn, hole_kappa)
    final_kappas = channels.final_kappas(hole_kappa, only_reachable=True)

    ekin = get_electron_kinetic_energy_Hartree(one_photon, n_qn, hole_kappa)
    coulomb_phase_arr = np.zeros(
        (len(final_kappas), len(ekin))
    )  # vector to store coulomb phase

    for i in range(len(final_kappas)):
        final_kappa = final_kappas[i]
        coulomb_phase_arr[i, :] = coulomb_phase(final_kappa, ekin, Z)

    return coulomb_phase_arr


def get_matrix_elements_with_coulomb_phase(one_photon: OnePhoton, n_qn, hole_kappa, Z):
    """
    Computes matrix elements for all possible final states of the given hole
    and adds Coulomb phase to them.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion

    Returns:
    Matrix elements with Coulomb phase
    """

    one_photon.assert_hole_load(n_qn, hole_kappa)

    M = get_matrix_elements_for_all_final_states(one_photon, n_qn, hole_kappa)
    coul_phase = get_coulomb_phase(one_photon, n_qn, hole_kappa, Z)  # Coulomb phase

    assert (
        M.shape == coul_phase.shape
    ), "Shapes of matrix with elements and matrix with Coulomb phase don't match!"

    return M * np.exp(1j * coul_phase)


def prepare_absorption_and_emission_matrices_1sim(
    one_photon: OnePhoton, n_qn, hole_kappa, Z, steps_per_IR_photon=None
):
    """
    Works with the case of 1 simulation (only 1 OnePhoton object). Constructs absorption and
    emission matrices (the same matrix in this case) and matches them so that they corrsepond
    to the same final photoelectron energies.

    Params:
    one_photon - object of the OnePhoton class with some loaded holes
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    steps_per_IR_photon - the number of XUV energy steps fitted in the IR photon energy.
    If not specified, the the program calculates it based on the XUV energy data in the
    omega.dat file and value of the IR photon energy

    Returns:
    ekin_final - array of final phototelctron kinetic energies in eV
    M_emi_matched - matrix elements for emission path matched to the final energies
    M_abs_matched - matrix elements for absorption path matched to the final energies
    """

    one_photon.assert_hole_load(n_qn, hole_kappa)

    ekin_eV = get_electron_kinetic_energy_eV(one_photon, n_qn, hole_kappa)

    g_omega_IR = one_photon.g_omega_IR  # frequncy of the IR photon (in Hartree)

    if not steps_per_IR_photon:
        steps_per_IR_photon = int(
            g_omega_IR / ((ekin_eV[1] - ekin_eV[0]) / g_eV_per_Hartree)
        )

    M = get_matrix_elements_with_coulomb_phase(one_photon, n_qn, hole_kappa, Z)
    ekin_final, M_emi_matched, M_abs_matched = (
        match_absorption_and_emission_matrices_1sim(ekin_eV, M, M, steps_per_IR_photon)
    )

    return ekin_final, M_emi_matched, M_abs_matched


def match_absorption_and_emission_matrices_1sim(
    energies, M_emi, M_abs, steps_per_IR_photon
):
    """
    Matches absorpiton and emission matrices so that their indices correspond to the same final
    photoelectron energy in the case of 1 simulation.

    Params:
    energies - array of energies
    M_emi - unmatched matrix elements for emission path
    M_abs - unmatched matrix elements for absorption path
    steps_per_IR_photon - the number of XUV energy steps fitted in the IR photon energy

    Returns:
    energies_final - the array of final phototelctron energies
    M_emi_matched - matrix elements for emission path matched to the final energies
    M_abs_matched - matrix elements for absorption path matched to the final energies
    """

    energies_final = final_energies_for_matching_1sim(energies, steps_per_IR_photon)

    assert (
        M_abs.shape[0] == M_emi.shape[0]
    ), "The number of final states in absorption and emission matrices is different!"

    M_emi_matched = np.zeros(
        (M_emi.shape[0], M_emi.shape[1] - 2 * steps_per_IR_photon),
        dtype="complex128",
    )

    M_abs_matched = np.zeros(
        (M_abs.shape[0], M_abs.shape[1] - 2 * steps_per_IR_photon),
        dtype="complex128",
    )

    for i in range(M_abs.shape[0]):
        M_emi_matched[i, :], M_abs_matched[i, :] = match_matrix_elements_1sim(
            M_emi[i, :], M_abs[i, :], steps_per_IR_photon
        )

    return energies_final, M_emi_matched, M_abs_matched


def prepare_absorption_and_emission_matrices_2sim(
    one_photon_emi: OnePhoton,
    one_photon_abs: OnePhoton,
    n_qn,
    hole_kappa,
    Z,
    energies_mode="both",
):
    """
    Works with the case of 2 simulations (requires 2 OnePhoton objects).
    Constructs emission from one_photon_1 and absorption from one_photon_2 matrices and
    matches them so that they corrsepond to the same final photoelectron energies.

    Params:
    one_photon_emi - object of the OnePhoton class for emission matrix
    one_photon_abs - object of the OnePhoton class for absorption matrix
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    energies_mode - tells which energies we take for matrices interpolation. Possible options:
    "emi" - energies from emission object, "abs" - energies from absorption object, "both" -
    combined array from both emission and absorption objects.

    Returns:
    ekin_final - array of final phototelctron kinetic energies in eV
    M_emi_matched - matrix elements for emission path matched to the final energies
    M_abs_matched - matrix elements for absorption path matched to the final energies
    """

    one_photon_emi.assert_hole_load(n_qn, hole_kappa)
    one_photon_abs.assert_hole_load(n_qn, hole_kappa)

    ekin_eV_emi = get_electron_kinetic_energy_eV(one_photon_emi, n_qn, hole_kappa)
    ekin_eV_abs = get_electron_kinetic_energy_eV(one_photon_abs, n_qn, hole_kappa)

    g_omega_IR_emi = one_photon_emi.g_omega_IR  # frequncy of the IR photon (in Hartree)
    g_omega_IR_abs = one_photon_abs.g_omega_IR

    # match to the final photoelectron energies
    ekin_eV_emi -= g_omega_IR_emi * g_eV_per_Hartree
    ekin_eV_abs += g_omega_IR_abs * g_eV_per_Hartree

    M_emi = get_matrix_elements_with_coulomb_phase(one_photon_emi, n_qn, hole_kappa, Z)
    M_abs = get_matrix_elements_with_coulomb_phase(one_photon_abs, n_qn, hole_kappa, Z)

    ekin_final, M_emi_matched, M_abs_matched = (
        match_absorption_and_emission_matrices_2sim(
            ekin_eV_emi, ekin_eV_abs, M_emi, M_abs, energies_mode=energies_mode
        )
    )

    return ekin_final, M_emi_matched, M_abs_matched


def match_absorption_and_emission_matrices_2sim(
    energies_emi,
    energies_abs,
    M_emi,
    M_abs,
    energies_mode="both",
):
    """
    Matches absorpiton and emission matrices so that their indices correspond to the same final
    photoelectron energy in the case of 2 simulations.

    Params:
    energies_emi - array of energies for emission path
    energies_abs - array of energies for absorption path
    M_emi - unmatched matrix elements for emission path
    M_abs - unmatched matrix elements for absorption path
    energies_mode - tells which energies we take for matrices interpolation. Possible options:
    "emi" - energies from emission object, "abs" - energies from absorption object, "both" -
    combined array from both emission and absorption objects.

    Returns:
    energies_final - the array of final phototelctron energies
    M_emi_matched - matrix elements for emission path matched to the final energies
    M_abs_matched - matrix elements for absorption path matched to the final energies
    """

    energies_final = final_energies_for_matching_2sim(
        energies_emi, energies_abs, energies_mode=energies_mode
    )

    assert (
        M_abs.shape[0] == M_emi.shape[0]
    ), "The number of final states in absorption and emission matrices is different!"

    M_emi_matched = np.zeros(
        (M_emi.shape[0], len(energies_final)),
        dtype="complex128",
    )

    M_abs_matched = np.zeros(
        (M_abs.shape[0], len(energies_final)),
        dtype="complex128",
    )
    for i in range(M_abs.shape[0]):
        M_emi_matched[i, :], M_abs_matched[i, :] = match_matrix_elements_2sim(
            energies_final, energies_emi, energies_abs, M_emi[i, :], M_abs[i, :]
        )

    return energies_final, M_emi_matched, M_abs_matched


def get_prepared_matrices(
    one_photon_1: OnePhoton,
    n_qn,
    hole_kappa,
    Z,
    one_photon_2: Optional[OnePhoton] = None,
    steps_per_IR_photon=None,
    energies_mode="both",
):
    """
    Returns prepared emission and absorption matrices as well as energy vector
    depending on how many simulations (1 or 2) were provided.

    Params:
    one_photon_1 - object of the OnePhoton class corresponding to one simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    one_photon_2 - second object of the OnePhoton class if we want to consider 2 simulations
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.

    Returns:
    ekin_eV - array of phototelctron kinetic energies in eV
    M_emi_matched - matrix elements for emission path matched to the final energies
    M_abs_matched - matrix elements for absorption path matched to the final energies
    """

    one_photon_1.assert_hole_load(n_qn, hole_kappa)

    if one_photon_2:  # if the second simulation is provided
        one_photon_2.assert_hole_load(n_qn, hole_kappa)
        ekin_eV, M_emi_matched, M_abs_matched = (
            prepare_absorption_and_emission_matrices_2sim(
                one_photon_1,
                one_photon_2,
                n_qn,
                hole_kappa,
                Z,
                energies_mode=energies_mode,
            )
        )
    else:  # if the second simulation is not provided
        ekin_eV, M_emi_matched, M_abs_matched = (
            prepare_absorption_and_emission_matrices_1sim(
                one_photon_1,
                n_qn,
                hole_kappa,
                Z,
                steps_per_IR_photon=steps_per_IR_photon,
            )
        )

    return ekin_eV, M_emi_matched, M_abs_matched
