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

        self.one_photon = (
            one_photon  # object of the OnePhotonInitializer with some initialized holes
        )

    def __assert_hole_initialization(self, hole_kappa):

        assert self.one_photon.is_initialized(
            hole_kappa
        ), f"The hole with kappa {hole_kappa} is not initialized!"

    def __assert_final_kappa(self, hole_kappa, final_kappa):

        assert self.one_photon.check_final_kappa(
            hole_kappa, final_kappa
        ), f"The final hole with kappa {final_kappa} is not within channels for the inital hole with kappa {hole_kappa}!"

    def get_omega_eV(self, hole_kappa):

        self.__assert_hole_initialization(hole_kappa)

        return self.get_omega_Hartree(hole_kappa) * g_eV_per_Hartree

    def get_omega_Hartree(self, hole_kappa):

        self.__assert_hole_initialization(hole_kappa)

        channel = self.one_photon.channels[hole_kappa]
        omega_Hartree = channel.raw_data[
            :, 0
        ]  # omega energies in Hartree from the output file.

        return omega_Hartree

    def get_electron_kinetic_energy_Hartree(self, hole_kappa):
        """
        Returns electron kinetic energy in Hartree.
        """

        self.__assert_hole_initialization(hole_kappa)

        return (
            self.get_omega_Hartree(hole_kappa)
            - self.one_photon.channels[hole_kappa].hole.binding_energy
        )

    def get_electron_kinetic_energy_eV(self, hole_kappa):
        """
        Returns electron kinetic energy in eV.
        """

        self.__assert_hole_initialization(hole_kappa)

        return self.get_electron_kinetic_energy_Hartree(hole_kappa) * g_eV_per_Hartree

    # TODO: Implement calculation of cross section via amplitudes.
    # TODO: Implement calculation of cross section via matrix elements.

    def get_partial_integrated_cross_section(
        self, hole_kappa, final_kappa, divide_omega=True
    ):
        # Depending on conventions when creating the dipole elements in the Fortran program we might
        # have to divide or multiply by the photon energy (omega) when calculating cross sections.
        # Usually it is correct to divide by omega, and that is default behaviour of this function.

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
        Sums over all partial integrated cross sections for all possible final states.
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
        emission energies). Parameter ekin_final allows specifiying custom array of kinetic energies
        to compute the cross section. If not specified, the function concatenates and sorts
        kinetic energy vectors for all holes.
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

        ekin, emission_cs = self.__interploate_photoelectron_emission_cross_section(
            N_holes, holes_ekin, holes_cs, ekin_final
        )

        return ekin, emission_cs

    @staticmethod
    def __interploate_photoelectron_emission_cross_section(
        N_holes, holes_ekin, holes_cs, ekin_final
    ):
        """
        Peforms linear interpolation of the photoelectron emission cross sections for different holes
        to match them for the same electron kinetic energy
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
        """Returns the value of the matrix element after one photon as amp*[e^(i*phase_of_F),
        e^(i*phase_of_G)]."""

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
        Computes matrix elements for all possible final states.
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
        Computes coulomb phase for all the final holes of the given hole.
        Z - charge of the ion.
        """

        self.__assert_hole_initialization(hole_kappa)

        channel = self.one_photon.channels[hole_kappa]
        final_kappas = channel.final_kappas(hole_kappa, only_reachable=True)

        ekin = self.get_electron_kinetic_energy_Hartree(hole_kappa)
        coulomb_phase_vec = np.zeros(
            (len(final_kappas), len(ekin))
        )  # vector to store coulomb phase

        for i in range(len(final_kappas)):
            final_kappa = final_kappas[i]
            coulomb_phase_vec[i, :] = coulomb_phase(final_kappa, ekin, Z)

        return coulomb_phase_vec

    def get_matrix_elements_with_coulomb_phase(self, hole_kappa, Z):
        """
        Computes matrix elements for all possible final states and adds Coulomb phase to them.
        Z - charge of the ion.
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
        """This function returns Wigner intensity for a photoelectron that has absorbed
        one photon. M_emi - matched matrix for emission path. M_abs - matched matrix for
        absorption path."""

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
        """This function returns the value of the asymmetry parameter for a state defined by hole_kappa in the one photon case.
        M1 and M2 contains the matrix elements and other phases of the wave function organized according to their final kappa like so:
        m = |hole_kappa|
        s = sign(hole_kappa)
        M = [s(m-1), -sm, s(m+1)]
        The formula for the asymmetry parameter has the form beta_2 = coeff(k1,k1)*M1(k1)*M2(k1)^* + coeff(k1,k2)*M1(k1)*M2(k2)^*
        + coeff(k1,k3)*M1(k1)*M2(k3)^* + coeff(k2,k1)*M1(k2)*M2(k1)^* + ... / (coeff(k1)M1(k1)M2(k2)^* + coeff(k2)M1(k2)M2(k2)^* + coeff(k3)M1(k3)M2(k3)^*)
        The two different input matrix elements correspond to the one photon matrix elements at different energies,
        for example two absorption and emission branches of a RABBIT experiment.
        If you want to calculate the parameters for only the absorption path, simply pass in Ma in both M1 and M2.
        If you want to use some other values for the coefficients used in the calculation than the default,
        set path = "path/to/folder/containing/coefficient/files".
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
        steps_per_IR_photon is the ratio between frequncy of IR photon and step size for XUV
        photon energies.
        """

        self.__assert_hole_initialization(hole_kappa)

        g_omega_IR = self.one_photon.channels[
            hole_kappa
        ].g_omega_IR  # frequncy of the IR photon (in Hartree)

        ekin_shifted, M_abs_shifted, M_emi_shifted = (
            self.__prepare_data_for_wigner_delay(
                hole_kappa, Z, g_omega_IR, steps_per_IR_photon
            )
        )

        tau_int_wigner = self.__integrated_wigner_delay_from_intensity(
            hole_kappa, M_emi_shifted, M_abs_shifted, g_omega_IR
        )

        return ekin_shifted, tau_int_wigner

    def get_integrated_wigner_phase(
        self, hole_kappa, Z, steps_per_IR_photon=None, unwrap=True
    ):
        """
        Computes integrated wigner phase from the wigner delay.
        Unwrap parameter tells if to unwrap phase using np.unwrap.
        """

        self.__assert_hole_initialization(hole_kappa)

        g_omega_IR = self.one_photon.channels[
            hole_kappa
        ].g_omega_IR  # frequncy of the IR photon (in Hartree)

        ekin_shifted, tau_int_wigner = self.get_integrated_wigner_delay(
            hole_kappa, Z, steps_per_IR_photon
        )
        phase_int_wigner = delay_to_phase(tau_int_wigner, g_omega_IR)

        if unwrap:
            phase_int_wigner = unwrap_phase_with_nans(phase_int_wigner)

        return ekin_shifted, phase_int_wigner

    def get_angular_wigner_delay(self, hole_kappa, Z, angle, steps_per_IR_photon=None):

        self.__assert_hole_initialization(hole_kappa)

        g_omega_IR = self.one_photon.channels[
            hole_kappa
        ].g_omega_IR  # frequncy of the IR photon (in Hartree)

        ekin_shifted, M_abs_shifted, M_emi_shifted = (
            self.__prepare_data_for_wigner_delay(
                hole_kappa, Z, g_omega_IR, steps_per_IR_photon
            )
        )

        tau_ang_wigner = self.__angular_wigner_delay_from_asymmetry_parameter(
            hole_kappa, M_emi_shifted, M_abs_shifted, g_omega_IR, angle
        )

        return ekin_shifted, tau_ang_wigner

    def get_angular_wigner_phase(
        self, hole_kappa, Z, angle, steps_per_IR_photon=None, unwrap=True
    ):
        """
        Computes angular part of wigner phase from the wigner delay.
        Unwrap parameter tells if to unwrap phase using np.unwrap.
        """

        self.__assert_hole_initialization(hole_kappa)

        g_omega_IR = self.one_photon.channels[
            hole_kappa
        ].g_omega_IR  # frequncy of the IR photon (in Hartree)

        ekin_shifted, tau_ang_wigner = self.get_angular_wigner_delay(
            hole_kappa,
            Z,
            angle,
            steps_per_IR_photon,
        )
        phase_ang_wigner = delay_to_phase(tau_ang_wigner, g_omega_IR)

        if unwrap:
            phase_ang_wigner = unwrap_phase_with_nans(phase_ang_wigner)

        return ekin_shifted, phase_ang_wigner

    def get_wigner_delay(self, hole_kappa, Z, angle, steps_per_IR_photon=None):
        """
        Computes total Wigner delay: integrated + angular parts
        """

        self.__assert_hole_initialization(hole_kappa)

        g_omega_IR = self.one_photon.channels[
            hole_kappa
        ].g_omega_IR  # frequncy of the IR photon (in Hartree)

        ekin_shifted, M_abs_shifted, M_emi_shifted = (
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

        return ekin_shifted, tau_wigner

    def get_wigner_phase(
        self, hole_kappa, Z, angle, steps_per_IR_photon=None, unwrap=True
    ):
        """
        Computes angular part of wigner phase from the wigner delay.
        Unwrap parameter tells if to unwrap phase using np.unwrap.
        """

        self.__assert_hole_initialization(hole_kappa)

        g_omega_IR = self.one_photon.channels[
            hole_kappa
        ].g_omega_IR  # frequncy of the IR photon (in Hartree)

        ekin_shifted, tau_wigner = self.get_wigner_delay(
            hole_kappa,
            Z,
            angle,
            steps_per_IR_photon,
        )
        phase_wigner = delay_to_phase(tau_wigner, g_omega_IR)

        if unwrap:
            phase_wigner = unwrap_phase_with_nans(phase_wigner)

        return ekin_shifted, phase_wigner

    def __prepare_data_for_wigner_delay(
        self, hole_kappa, Z, g_omega_IR, steps_per_IR_photon
    ):
        """
        Prepares matrices and energy vector for wigner delay computations.
        Constructs steps_per_IR_photon if not specified and matrix elements, shifts
        matrix elements and energy vectors to match final photoelectron energies.
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
        self, hole_kappa, M_emi, M_abs, g_omega_IR
    ):
        """
        Computes integrated Wigner delay from Wigner intenisty
        """

        wigner_intensity = self.get_wigner_intensity(hole_kappa, M_emi, M_abs)

        tau_int_wigner = (
            g_inverse_atomic_frequency_to_attoseconds
            * np.angle(wigner_intensity)
            / (2.0 * g_omega_IR)
        )

        return tau_int_wigner

    def __angular_wigner_delay_from_asymmetry_parameter(
        self, hole_kappa, M_emi, M_abs, g_omega_IR, angle
    ):
        """
        Computes angular part of Wigner delay from the complex assymetry parameter.
        """

        b2_complex, _ = self.get_one_photon_asymmetry_parameter(
            hole_kappa, M_emi, M_abs, "cross"
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
        """Shifts the matrix element and energy arrays so that the same index in all
        of them corresponds to the same final photoelectron energy"""

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
