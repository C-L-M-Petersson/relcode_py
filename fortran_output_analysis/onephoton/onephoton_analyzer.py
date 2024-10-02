import numpy as np
import os
import math
from scipy.special import legendre
from fortran_output_analysis.constants_and_parameters import (
    g_eV_per_Hartree,
    g_inverse_atomic_frequency_to_attoseconds,
)
from fortran_output_analysis.common_utility import (
    convert_rate_to_cross_section,
    exported_mathematica_tensor_to_python_list,
    coulomb_phase,
    delay_to_phase,
    unwrap_phase_with_nans,
)
from fortran_output_analysis.onephoton.onephoton import OnePhoton


class OnePhotonAnalyzer:
    """
    Takes initialized one photon object as input and contains methods to analyze its
    raw output data.
    """

    def __init__(self, one_photon: OnePhoton):
        """
        Params:
        one_photon - object of the OnePhoton class with some initialized holes
        """

        self.one_photon = one_photon

    def __assert_hole_initialization(self, hole_kappa):
        """
        Assertion of the hole initialization.

        Params:
        hole_kappa - kappa value of the hole
        """

        assert self.one_photon.is_initialized(
            hole_kappa
        ), f"The hole with kappa {hole_kappa} is not initialized!"

    def __assert_final_kappa(self, hole_kappa, final_kappa):
        """
        Assertion of the final state. Checks if the given final state
        is within possible ionization channels for the given hole.

        Params:
        hole_kappa - kappa value of the hole
        final_kappa - kappa value of the final state
        """

        assert self.one_photon.check_final_kappa(
            hole_kappa, final_kappa
        ), f"The final state with kappa {final_kappa} is not within channels for the inital hole with kappa {hole_kappa}!"

    def get_omega_eV(self, hole_kappa):
        """
        Returns array of XUV photon energies in eV for the given hole.

        Params:
        hole_kappa - kappa value of the hole

        Returns:
        array of XUV photon energies in eV for the given hole
        """

        self.__assert_hole_initialization(hole_kappa)

        return self.get_omega_Hartree(hole_kappa) * g_eV_per_Hartree

    def get_omega_Hartree(self, hole_kappa):
        """
        Returns array of XUV photon energies in Hartree for the given hole.

        Params:
        hole_kappa - kappa value of the hole

        Returns:
        omega_Hartree - array of XUV photon energies in Hartree for the given hole
        """

        self.__assert_hole_initialization(hole_kappa)

        channel = self.one_photon.channels[hole_kappa]
        omega_Hartree = channel.raw_data[
            :, 0
        ]  # omega energies in Hartree from the output file.

        return omega_Hartree

    def get_electron_kinetic_energy_Hartree(self, hole_kappa):
        """
        Returns array of electron kinetic energies in Hartree for the given hole.

        Params:
        hole_kappa - kappa value of the hole

        Returns:
        array of electron kinetic energies in Hartree for the given hole
        """

        self.__assert_hole_initialization(hole_kappa)

        return (
            self.get_omega_Hartree(hole_kappa)
            - self.one_photon.channels[hole_kappa].hole.binding_energy
        )

    def get_electron_kinetic_energy_eV(self, hole_kappa):
        """
        Returns array of electron kinetic energies in eV for the given hole.

        Params:
        hole_kappa - kappa value of the hole

        Returns:
        array of electron kinetic energies in eV for the given hole
        """

        self.__assert_hole_initialization(hole_kappa)

        return self.get_electron_kinetic_energy_Hartree(hole_kappa) * g_eV_per_Hartree

    # TODO: Implement calculation of cross section via amplitudes.
    # TODO: Implement calculation of cross section via matrix elements.

    def get_partial_integrated_cross_section(
        self, hole_kappa, final_kappa, divide_omega=True
    ):
        """
        Calculates integrated partial cross section: integrated cross section for only one
        ionization channel (final state) of the given hole.
        Depending on conventions when creating the dipole elements in the Fortran program we
        might have to divide or multiply by the photon energy (omega) when calculating
        cross sections. Usually it is correct to divide by omega, and that is default behaviour
        of this function.

        Params:
        hole_kappa - kappa value of the hole
        final_kappa - kappa value of the final state
        divide_omega - tells if we divide or multiply by the photon energy (omega) when
        calculating the cross section

        Returns:
        ekin_eV - array of electron kinetic energy
        cross_section - values of the partial integrated cross section
        """

        self.__assert_final_kappa(hole_kappa, final_kappa)

        channel = self.one_photon.channels[hole_kappa]
        rate = channel.get_rate_for_channel(final_kappa)
        omega = self.get_omega_Hartree(hole_kappa)
        cross_section = convert_rate_to_cross_section(rate, omega, divide_omega)

        ekin_eV = self.get_electron_kinetic_energy_eV(
            hole_kappa
        )  # electron kinetic energy in eV

        return ekin_eV, cross_section

    def get_total_integrated_cross_section_for_hole(
        self, hole_kappa, divide_omega=True
    ):
        """
        Calculates total integrated cross section: sums over all partial integrated cross
        sections for the give hole.

        Params:
        hole_kappa - kappa value of the hole
        divide_omega - tells if we divide or multiply by the photon energy (omega) when
        calculating the cross section

        Returns:
        ekin_eV - array of electron kinetic energy
        total_cs - values of the total integrated cross section
        """

        self.__assert_hole_initialization(hole_kappa)

        # Use electron kinetic energy to initialise total cs array:
        ekin_eV = self.get_electron_kinetic_energy_eV(
            hole_kappa
        )  # electron kinetic energy in eV

        N = len(ekin_eV)
        total_cs = np.zeros(N)
        channel = self.one_photon.channels[hole_kappa]
        for final_key in channel.final_states.keys():
            final_state = channel.final_states[final_key]
            final_kappa = final_state.kappa
            _, patrial_cs = self.get_partial_integrated_cross_section(
                hole_kappa, final_kappa, divide_omega
            )
            total_cs += patrial_cs

        return ekin_eV, total_cs

    def get_integrated_photon_absorption_cross_section(self):
        """
        Calculates integrated photon absroption cross section: sums total integrated
        cross sections for all initialized holes.

        Returns:
        omega_eV - array of photon energy
        absorption_cs - values of the photon absroption cross section
        """

        # TODO: add case when different holes have different photon energies

        initialized_holes = list(self.one_photon.channels.keys())

        assert (
            len(initialized_holes) > 0
        ), "No holes are initialized. Please, initialize at least one hole!"

        first_hole_kappa = initialized_holes[0]
        omega_eV = self.get_omega_eV(first_hole_kappa)  # XUV photon energy in eV
        N = len(omega_eV)
        absorption_cs = np.zeros(N)

        for hole_kappa in initialized_holes:
            _, total_cs = self.get_total_integrated_cross_section_for_hole(hole_kappa)
            absorption_cs += total_cs

        return omega_eV, absorption_cs

    def get_integrated_photoelectron_emission_cross_section(self, ekin_final=None):
        """
        Computes photoelectron emission cross section (cross section for photoelectron
        emission energies): sums total integrated cross sections for all initialized holes and
        interpolates them to the same photoelectron kinetic energy.

        Params:
        ekin_final - allows specifiying custom array of photoelectron kinetic energies
        to compute the cross section. If not specified, the function concatenates and sorts
        kinetic energy vectors for all holes

        Returns:
        ekin_eV - array of final photoelectron kinetic energies in eV
        emission_cs - values of the interpolated photonelctron emission cross section
        """

        initialized_holes = list(self.one_photon.channels.keys())

        assert (
            len(initialized_holes) > 0
        ), "No holes are initialized. Please, initialize at least one hole!"

        first_hole_kappa = initialized_holes[0]
        ekin_eV_first = self.get_electron_kinetic_energy_eV(
            first_hole_kappa
        )  # electron kinetic energy in eV for the first hole
        N_ekin = len(ekin_eV_first)  # length of the electron kinetic energy vetor
        N_holes = len(initialized_holes)  # total number of holes

        holes_ekin = (
            []
        )  # list to store photoelectron kinetic energies for different holes
        holes_cs = np.zeros(
            (N_holes, N_ekin)
        )  # array to store total cross sections for different holes

        for i in range(N_holes):
            hole_kappa = initialized_holes[i]
            ekin, hole_cs = self.get_total_integrated_cross_section_for_hole(hole_kappa)
            holes_ekin.append(ekin)

            holes_cs[i, :] = hole_cs

        ekin_eV, emission_cs = self.__interploate_photoelectron_emission_cross_section(
            N_holes, holes_ekin, holes_cs, ekin_final
        )

        return ekin_eV, emission_cs

    @staticmethod
    def __interploate_photoelectron_emission_cross_section(
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

    def get_matrix_elements_for_final_state(self, hole_kappa, final_kappa):
        """
        Computes matrix elements after one photon as amp*[e^(i*phase_of_F),
        e^(i*phase_of_G)] for the given hole and final state.

        Params:
        hole_kappa - kappa value of the given hole
        final_kappa - kappa value of the final state

        Returns:
        matrix elements after one photon
        """

        self.__assert_final_kappa(hole_kappa, final_kappa)

        channel = self.one_photon.channels[hole_kappa]
        final_state = channel.final_states[final_kappa]
        # We assume that the data is sorted the same in amp_all and phaseF_all as in pcur_all
        # this is true at time of writing (2022-05-23).
        column_index = final_state.pcur_column_index
        return channel.raw_amp_data[:, column_index] * [
            np.exp(1j * channel.raw_phaseF_data[:, column_index]),
            np.exp(1j * channel.raw_phaseG_data[:, column_index]),
        ]

    def get_matrix_elements_for_all_final_states(self, hole_kappa):
        """
        Computes matrix elements for all possible final states of the given hole.

        Params:
        hole_kappa - kappa value of the given hole

        Returns:
        M - matrix elements
        """

        self.__assert_hole_initialization(hole_kappa)

        channel = self.one_photon.channels[hole_kappa]
        final_kappas = channel.final_kappas(hole_kappa, only_reachable=True)

        # the first kappa from the final_kappas list
        first_of_final_kappas = final_kappas[0]

        # [0] since we are only interested in the largest relativistic component
        matrix_elements = self.get_matrix_elements_for_final_state(
            hole_kappa, first_of_final_kappas
        )[0]

        M = np.zeros(
            (len(final_kappas), len(matrix_elements)), dtype="complex128"
        )  # initialize the matrix
        M[0, :] = matrix_elements  # put the matrix elements for the first kappa

        for i in range(1, len(final_kappas)):
            final_kappa = final_kappas[i]
            M[i, :] = self.get_matrix_elements_for_final_state(hole_kappa, final_kappa)[
                0
            ]

        return M

    def get_coulomb_phase(self, hole_kappa, Z):
        """
        Computes Coulomb phase for all the final states of the given hole.

        Params:
        hole_kappa - kappa value of the given hole
        Z - charge of the ion

        Returns:
        coulomb_phase_arr - array with Coulomb phases
        """

        self.__assert_hole_initialization(hole_kappa)

        channel = self.one_photon.channels[hole_kappa]
        final_kappas = channel.final_kappas(hole_kappa, only_reachable=True)

        ekin = self.get_electron_kinetic_energy_Hartree(hole_kappa)
        coulomb_phase_arr = np.zeros(
            (len(final_kappas), len(ekin))
        )  # vector to store coulomb phase

        for i in range(len(final_kappas)):
            final_kappa = final_kappas[i]
            coulomb_phase_arr[i, :] = coulomb_phase(final_kappa, ekin, Z)

        return coulomb_phase_arr

    def get_matrix_elements_with_coulomb_phase(self, hole_kappa, Z):
        """
        Computes matrix elements for all possible final states of the given hole
        and adds Coulomb phase to them.

        Params:
        hole_kappa - kappa value of the given hole
        Z - charge of the ion

        Returns:
        Matrix elements with Coulomb phase
        """

        self.__assert_hole_initialization(hole_kappa)

        M = self.get_matrix_elements_for_all_final_states(hole_kappa)
        coul_phase = self.get_coulomb_phase(hole_kappa, Z)  # Coulomb phase

        assert (
            M.shape == coul_phase.shape
        ), "Shapes of matrix with elements and matrix with Coulomb phase don't match!"

        return M * np.exp(1j * coul_phase)

    def get_wigner_intensity(
        self,
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
        hole_kappa - kappa value of the hole
        M_emi - matrix for emission path matched to the same final photoelectron
        energies
        M_abs - matrix for absorption path matched to the same final photoelectron
        energies
        path - path to the file with coefficients for Wigner intensity calculation

        Returns:
        wigner_intensity - array with Wigner intensity values
        """

        self.__assert_hole_initialization(hole_kappa)

        assert (
            M_emi.shape == M_abs.shape
        ), "The shapes of the input matrices must be the same!"

        length = M_emi.shape[1]

        if path[-1] is not os.path.sep:
            path = path + os.path.sep

        try:
            with open(
                path + f"integrated_intensity_{hole_kappa}.txt", "r"
            ) as coeffs_file:
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

    def get_one_photon_asymmetry_parameter(
        self,
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
        original matrix to emission and absorption paths
        (using e.g. self.__match_matrix_elements_and_energies_to_same_final_photoelectron_energy)
        and basically get two different matrices. Then pass the matrix matched for emission
        path as M1 and the one matched for absorption path as M2 and
        specify abs_emi_or_cross as "cross".

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

        self.__assert_hole_initialization(hole_kappa)

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

        assert (
            M1.shape == M2.shape
        ), "The shapes of the input matrices must be the same!"

        data_size = M1.shape[1]

        # Try opening the needed file.
        try:
            with open(
                path + f"asymmetry_coeffs_2_{hole_kappa}.txt", "r"
            ) as coeffs_file:
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

    def get_angular_part_of_cross_section(self, hole_kappa, Z, angle):
        """
        Computes angular part of the total cross section for the given hole.

        Params:
        hole_kappa - kappa value of the hole
        Z - charge of the ion
        angle - angle to compute cross section

        Returns:
        ekin_eV - array of photoelectron kinetic energy in eV
        angular_part - angular part of the cross section
        """

        self.__assert_hole_initialization(hole_kappa)

        ekin_eV = self.get_electron_kinetic_energy_eV(hole_kappa)
        M = self.get_matrix_elements_with_coulomb_phase(hole_kappa, Z)

        b2_real, _ = self.get_one_photon_asymmetry_parameter(
            hole_kappa, M, M, "abs"
        )  # one-photon real assymetry parameter

        angular_part = 1 + b2_real * legendre(2)(
            np.array(np.cos(math.radians(angle)))
        )  # angluar part of the cross section

        return ekin_eV, angular_part

    def get_total_cross_section_for_hole(self, hole_kappa, Z, angle):
        """
        Computes total cross section (integrated part * angular part) for the given hole and
        given angle.

        Params:
        hole_kappa - kappa value of the hole
        Z - charge of the ion
        angle - angle to compute cross section

        Returns:
        ekin_eV - array of photoelectron kinetic energy in eV
        angular_part - total cross section
        """

        _, integrated_part = self.get_total_integrated_cross_section_for_hole(
            hole_kappa, divide_omega=True
        )
        ekin_eV, angular_part = self.get_angular_part_of_cross_section(
            hole_kappa, Z, angle
        )

        return ekin_eV, integrated_part * angular_part

    def get_integrated_wigner_delay(self, hole_kappa, Z, steps_per_IR_photon=None):
        """
        Computes integrated wigner delay for the given hole.

        Params:
        hole_kappa - kappa value of the hole
        Z - charge of the ion
        steps_per_IR_photon - the number of XUV energy steps fitted in the IR photon energy.
        If not specified, the the program calculates it based on the XUV energy data in the
        omega.dat file and initialized value of the IR photon energy.

        Returns:
        ekin_shifted_eV - photoelectron kinetic energy shifted to match the same final energy
        tau_int_wigner - array with integrated Wigner delays
        """

        self.__assert_hole_initialization(hole_kappa)

        g_omega_IR = self.one_photon.channels[
            hole_kappa
        ].g_omega_IR  # frequncy of the IR photon (in Hartree)

        ekin_shifted_eV, M_abs_shifted, M_emi_shifted = (
            self.__prepare_data_for_wigner_delay(
                hole_kappa, Z, g_omega_IR, steps_per_IR_photon
            )
        )

        tau_int_wigner = self.__integrated_wigner_delay_from_intensity(
            hole_kappa, M_emi_shifted, M_abs_shifted, g_omega_IR
        )

        return ekin_shifted_eV, tau_int_wigner

    def get_integrated_wigner_phase(
        self, hole_kappa, Z, steps_per_IR_photon=None, unwrap=True
    ):
        """
        Computes integrated wigner phase from the wigner delay.

        Params:
        hole_kappa - kappa value of the hole
        Z - charge of the ion
        steps_per_IR_photon - the number of XUV energy steps fitted in the IR photon energy.
        If not specified, the the program calculates it based on the XUV energy data in the
        omega.dat file and initialized value of the IR photon energy
        unwrap - if to unwrap phase using np.unwrap

        Returns:
        ekin_shifted_eV - photoelectron kinetic energy shifted to match the same final energy
        phase_int_wigner - array with integrated Wigner phases
        """

        self.__assert_hole_initialization(hole_kappa)

        g_omega_IR = self.one_photon.channels[
            hole_kappa
        ].g_omega_IR  # frequncy of the IR photon (in Hartree)

        ekin_shifted_eV, tau_int_wigner = self.get_integrated_wigner_delay(
            hole_kappa, Z, steps_per_IR_photon
        )
        phase_int_wigner = delay_to_phase(tau_int_wigner, g_omega_IR)

        if unwrap:
            phase_int_wigner = unwrap_phase_with_nans(phase_int_wigner)

        return ekin_shifted_eV, phase_int_wigner

    def get_angular_wigner_delay(self, hole_kappa, Z, angle, steps_per_IR_photon=None):
        """
        Computes angular part of the wigner delay.

        Params:
        hole_kappa - kappa value of the hole
        Z - charge of the ion
        angle - angle to compute the delay
        steps_per_IR_photon - the number of XUV energy steps fitted in the IR photon energy.
        If not specified, the the program calculates it based on the XUV energy data in the
        omega.dat file and initialized value of the IR photon energy

        Returns:
        ekin_shifted_eV - photoelectron kinetic energy shifted to match the same final energy
        tau_ang_wigner - array with angular part of Wigner delays
        """

        self.__assert_hole_initialization(hole_kappa)

        g_omega_IR = self.one_photon.channels[
            hole_kappa
        ].g_omega_IR  # frequncy of the IR photon (in Hartree)

        ekin_shifted_eV, M_abs_shifted, M_emi_shifted = (
            self.__prepare_data_for_wigner_delay(
                hole_kappa, Z, g_omega_IR, steps_per_IR_photon
            )
        )

        tau_ang_wigner = self.__angular_wigner_delay_from_asymmetry_parameter(
            hole_kappa, M_emi_shifted, M_abs_shifted, g_omega_IR, angle
        )

        return ekin_shifted_eV, tau_ang_wigner

    def get_angular_wigner_phase(
        self, hole_kappa, Z, angle, steps_per_IR_photon=None, unwrap=True
    ):
        """
        Computes angular part of Wigner phase from Wigner delay.

        Params:
        hole_kappa - kappa value of the hole
        Z - charge of the ion
        angle - angle to compute the phase
        steps_per_IR_photon - the number of XUV energy steps fitted in the IR photon energy.
        If not specified, the the program calculates it based on the XUV energy data in the
        omega.dat file and initialized value of the IR photon energy
        unwrap - if to unwrap phase using np.unwrap

        Returns:
        ekin_shifted_eV - photoelectron kinetic energy shifted to match the same final energy
        phase_ang_wigner - array with angular part of Wigner phases
        """

        self.__assert_hole_initialization(hole_kappa)

        g_omega_IR = self.one_photon.channels[
            hole_kappa
        ].g_omega_IR  # frequncy of the IR photon (in Hartree)

        ekin_shifted_eV, tau_ang_wigner = self.get_angular_wigner_delay(
            hole_kappa,
            Z,
            angle,
            steps_per_IR_photon,
        )
        phase_ang_wigner = delay_to_phase(tau_ang_wigner, g_omega_IR)

        if unwrap:
            phase_ang_wigner = unwrap_phase_with_nans(phase_ang_wigner)

        return ekin_shifted_eV, phase_ang_wigner

    def get_wigner_delay(self, hole_kappa, Z, angle, steps_per_IR_photon=None):
        """
        Computes total Wigner delay: integrated + angular part.

        Params:
        hole_kappa - kappa value of the hole
        Z - charge of the ion
        angle - angle to compute the delay
        steps_per_IR_photon - the number of XUV energy steps fitted in the IR photon energy.
        If not specified, the the program calculates it based on the XUV energy data in the
        omega.dat file and initialized value of the IR photon energy

        Returns:
        ekin_shifted_eV - photoelectron kinetic energy shifted to match the same final energy
        tau_wigner - array with total Wigner delays
        """

        self.__assert_hole_initialization(hole_kappa)

        g_omega_IR = self.one_photon.channels[
            hole_kappa
        ].g_omega_IR  # frequncy of the IR photon (in Hartree)

        ekin_shifted_eV, M_abs_shifted, M_emi_shifted = (
            self.__prepare_data_for_wigner_delay(
                hole_kappa, Z, g_omega_IR, steps_per_IR_photon
            )
        )

        tau_int_wigner = self.__integrated_wigner_delay_from_intensity(
            hole_kappa, M_emi_shifted, M_abs_shifted, g_omega_IR
        )

        tau_ang_wigner = self.__angular_wigner_delay_from_asymmetry_parameter(
            hole_kappa, M_emi_shifted, M_abs_shifted, g_omega_IR, angle
        )

        tau_wigner = tau_int_wigner + tau_ang_wigner  # total Wigner delay

        return ekin_shifted_eV, tau_wigner

    def get_wigner_phase(
        self, hole_kappa, Z, angle, steps_per_IR_photon=None, unwrap=True
    ):
        """
        Computes total Wigner phase: integrated + angular part.

        Params:
        hole_kappa - kappa value of the hole
        Z - charge of the ion
        angle - angle to compute the phase
        steps_per_IR_photon - the number of XUV energy steps fitted in the IR photon energy.
        If not specified, the the program calculates it based on the XUV energy data in the
        omega.dat file and initialized value of the IR photon energy

        Returns:
        ekin_shifted_eV - photoelectron kinetic energy shifted to match the same final energy
        phase_wigner - array with total Wigner phases
        """

        self.__assert_hole_initialization(hole_kappa)

        g_omega_IR = self.one_photon.channels[
            hole_kappa
        ].g_omega_IR  # frequncy of the IR photon (in Hartree)

        ekin_shifted_eV, tau_wigner = self.get_wigner_delay(
            hole_kappa,
            Z,
            angle,
            steps_per_IR_photon,
        )
        phase_wigner = delay_to_phase(tau_wigner, g_omega_IR)

        if unwrap:
            phase_wigner = unwrap_phase_with_nans(phase_wigner)

        return ekin_shifted_eV, phase_wigner

    def __prepare_data_for_wigner_delay(
        self, hole_kappa, Z, g_omega_IR, steps_per_IR_photon
    ):
        """
        Prepares matrices and energy vector for wigner delay computations.
        Constructs steps_per_IR_photon if not specified and matrix elements, shifts
        matrix elements and energy vector to match final photoelectron energies.

        Params:
        hole_kappa - kappa value of the hole
        Z - charge of the ion
        g_omega_IR - energy of the IR photon in Hartree
        steps_per_IR_photon - the number of XUV energy steps fitted in the IR photon energy.
        If not specified, the the program calculates it based on the XUV energy data in the
        omega.dat file and initialized value of the IR photon energy

        Returns:
        ekin_shifted - array of photoelectron energies shifted to match the same final energies
        M_abs_shifted - matrix elements for absorption path shifted to match the same
        final energies
        M_emi_shifted - matrix elements for emission path shifted to match the same
        final energies
        """

        ekin_eV = self.get_electron_kinetic_energy_eV(hole_kappa)

        if not steps_per_IR_photon:
            steps_per_IR_photon = int(
                g_omega_IR / ((ekin_eV[1] - ekin_eV[0]) / g_eV_per_Hartree)
            )

        M = self.get_matrix_elements_with_coulomb_phase(hole_kappa, Z)
        ekin_shifted, M_abs_shifted, M_emi_shifted = (
            self.__match_matrix_elements_and_energies_to_same_final_photoelectron_energy(
                ekin_eV, M, M, steps_per_IR_photon
            )
        )

        return ekin_shifted, M_abs_shifted, M_emi_shifted

    def __integrated_wigner_delay_from_intensity(
        self, hole_kappa, M_emi_shifted, M_abs_shifted, g_omega_IR
    ):
        """
        Computes integrated Wigner delay from Wigner intenisty.

        Params:
        hole_kappa - kappa value of the hole
        M_emi_shifted - matrix elements for emission path shifted to match the same
        final energies
        M_abs_shifted - matrix elements for absorption path shifted to match the same
        final energies
        g_omega_IR - energy of the IR photon in Hartree

        Returns:
        tau_int_wigner - array with integrated Wigner delay
        """

        wigner_intensity = self.get_wigner_intensity(
            hole_kappa, M_emi_shifted, M_abs_shifted
        )

        tau_int_wigner = (
            g_inverse_atomic_frequency_to_attoseconds
            * np.angle(wigner_intensity)
            / (2.0 * g_omega_IR)
        )

        return tau_int_wigner

    def __angular_wigner_delay_from_asymmetry_parameter(
        self, hole_kappa, M_emi_shifted, M_abs_shifted, g_omega_IR, angle
    ):
        """
        Computes angular part of Wigner delay from the complex assymetry parameter.

        Params:
        hole_kappa - kappa value of the hole
        M_emi_shifted - matrix elements for emission path shifted to match the same
        final energies
        M_abs_shifted - matrix elements for absorption path shifted to match the same
        final energies
        g_omega_IR - energy of the IR photon in Hartree
        angle - angle to compute delay

        Returns:
        tau_ang_wigner - array with angular part of Wigner delay
        """

        b2_complex, _ = self.get_one_photon_asymmetry_parameter(
            hole_kappa, M_emi_shifted, M_abs_shifted, "cross"
        )  # complex assymetry parameter for one photon case

        tau_ang_wigner = (
            g_inverse_atomic_frequency_to_attoseconds
            * np.angle(
                1.0 + b2_complex * legendre(2)(np.array(np.cos(math.radians(angle))))
            )
            / (2 * g_omega_IR)
        )

        return tau_ang_wigner

    @staticmethod
    def __match_matrix_elements_and_energies_to_same_final_photoelectron_energy(
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
        omega.dat file and initialized value of the IR photon energy

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
