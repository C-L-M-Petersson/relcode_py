import os
import numpy as np

from fortran_output_analysis.common_utility import (
    l_from_kappa,
    j_from_kappa,
    j_from_kappa_int,
    IonHole,
    load_raw_data,
    l_to_str,
    construct_hole_name,
)


class FinalState:
    """
    Stores inforation about final state in an inonization channel.
    """

    def __init__(self, kappa, pcur_col_idx):
        """
        Params:
        kappa - kappa value of the final state
        pcur_col_idx - index of the column in pcur_all.dat file corresponding to this
        final state
        """
        self.kappa = kappa
        self.l = l_from_kappa(kappa)
        self.j = j_from_kappa(kappa)
        self.name = l_to_str(self.l) + ("_{%i/2}" % (j_from_kappa_int(kappa)))
        self.pcur_column_index = pcur_col_idx


class Channels:
    """
    Stores inforation about ionization channels for the given hole.
    """

    def __init__(
        self,
        path_to_pcur,
        path_to_amp_all,
        path_to_phaseF_all,
        path_to_phaseG_all,
        hole_kappa,
        n_qn,
        binding_energy,
        g_omega_IR,
    ):
        """
        Params:
        path_to_pcur_all - path to file with probabilty current for one photon
        path_to_amp_all - path to file with amplitudes for one photon
        path_to_phaseF_all - path to file with the phase for larger relativistic component
        of the wave function
        path_to_phaseG_all - path to file with the phase for smaller relativistic component
        of the wave function
        hole_kappa - kappa value of the hole
        n_qn - principal quantum number of the hole
        binding_energy - binding energy of the hole
        g_omega_IR - energy of IR photon used in Fortaran simulations (in Hartree units)
        """

        self.path_to_pcur = path_to_pcur
        self.hole = IonHole(hole_kappa, n_qn, binding_energy)
        self.final_states = {}
        self.raw_data = load_raw_data(path_to_pcur)
        self.raw_amp_data = load_raw_data(path_to_amp_all)
        self.raw_phaseF_data = load_raw_data(path_to_phaseF_all)
        self.raw_phaseG_data = load_raw_data(path_to_phaseG_all)
        self.add_final_states()
        self.g_omega_IR = g_omega_IR  # energy of the IR photon (in Hartree)

    def add_final_states(self):
        """
        Adds final states of the hole's ionization channels. Excludes forbidden channels with
        kappa = 0.
        """
        kappa_hole = self.hole.kappa
        # One can convince oneself that the following is true for a given hole_kappa.
        #       possible_final_kappas = np.array([-kappa_hole, kappa_hole+1, -(-kappa_hole+1)])
        # It is possible that one of the final kappas are zero, so we need to handle this.
        # NOTE(anton): The pcur-files have three columns, one for each possible final kappa.
        # If there is no possibility for one of them the column is zero, and I
        # think the convention is that the zero column is the left-most (lowest index) then.
        # So if the kappas are sorted by ascending absolute value we should get this, since
        # if kappa = 0 the channel is closed.
        #       sort_kappas_idx = np.argsort(np.abs(possible_final_kappas))
        #       possible_final_kappas = possible_final_kappas[sort_kappas_idx]

        # This code should reproduce the previous implementation
        possible_final_kappas = self.final_kappas(kappa_hole, only_reachable=False)

        # This is for getting the data from the pcur files. The first column is the photon energy.
        pcur_column_index = 1
        for kappa in possible_final_kappas:
            if kappa != 0:
                self.final_states[kappa] = FinalState(kappa, pcur_column_index)

            pcur_column_index += 1

    def get_rate_for_channel(self, final_kappa):
        """
        Extracts raw rate data from the Fortran output file for the given final state.

        Params:
        final_kappa - kappa value of the final state

        Returns:
        rate - array with the raw rate data for the given final state
        """
        state = self.final_states[final_kappa]
        column_index = state.pcur_column_index
        rate = self.raw_data[:, column_index]
        return rate

    @staticmethod
    def final_kappas(hole_kappa, only_reachable=True):
        """
        Returns the possible final kappas that can be reached with one photon from
        an initial state with the given kappa. If only_reachable is False, this function
        will always return a list of three elements, even if one of them is 0.

        Params:
        hole_kappa - kappa value of the hole
        only_reachable - tells if only permitted states should be returned

        Returns:
        kappas - list with kappa values of possible final states
        """
        mag = np.abs(hole_kappa)
        sig = np.sign(hole_kappa)

        kappas = [sig * (mag - 1), -sig * mag, sig * (mag + 1)]

        if only_reachable:
            # Filter out any occurence of final kappa = 0
            kappas = [kappa for kappa in kappas if kappa != 0]

        return kappas


# TODO: probabily rewrite OnePhoton initialization so that it takes IonHole object as input


class OnePhoton:
    """
    Initializes holes for one photon case and reads raw data from the Fortran output.
    """

    def __init__(self, atom_name):
        # attributes for diag data
        self.diag_eigenvalues = np.array([])
        self.diag_matrix_elements = np.array([])
        self.__diag_initialized = False

        # attributes for holes' data
        self.name = atom_name
        self.channels = {}
        self.num_channels = 0

    def initialize_diag(
        self,
        path_to_diag_data,
        path_to_diag_eigenvalues=None,
        path_to_diag_matrix_elements=None,
        should_reinitialize=False,
    ):
        """
        Loads diagonal data: eigenvalues and matrix elements, and saves them
        to the corresponding object attributes: self.diag_eigenvalues and
        self.diag_matrix_elements.

        Params:
        path_to_diag_data - path to diagonal data directory
        path_to_diag_eigenvalues - path to the file with diagonal eigenvalues
        path_to_diag_matrix_elements - path to the file with diagonal eigenvalues
        should_reinitialize - tells whether we should reinitialize diagonal data if they
        were previously initialized
        """

        if not self.__diag_initialized or should_reinitialize:
            if self.__diag_initialized and should_reinitialize:
                print(
                    f"Reinitialize diagonal matrix elements and eigenvalues in {self.name}!"
                )

            if not path_to_diag_eigenvalues:
                path_to_diag_eigenvalues = (
                    path_to_diag_data + "diag_eigenvalues_Jtot1.dat"
                )
            if not path_to_diag_matrix_elements:
                path_to_diag_matrix_elements = (
                    path_to_diag_data + "diag_matrix_elements_Jtot1.dat"
                )

            self._load_diag_eigenvalues(path_to_diag_eigenvalues)
            self._load_diag_matrix_elements(path_to_diag_matrix_elements)
            self.__diag_initialized = True

    def _load_diag_eigenvalues(self, path_to_diag_eigenvalues):
        """
        Loads diagonal eigenvalues and saves them to self.diag_eigenvalues.

        Params:
        path_to_diag_eigenvalues - path to the file with diagonal eigenvalues
        """

        eigenvals_raw = np.loadtxt(path_to_diag_eigenvalues)
        eigvals_re = eigenvals_raw[:, 0]  # real part
        eigvals_im = eigenvals_raw[:, 1]  # imaginary part
        self.diag_eigenvalues = eigvals_re + 1j * eigvals_im

    def _load_diag_matrix_elements(self, path_to_diag_matrix_elements):
        """
        Loads diagonal matrix elements and saves them to self.diag_matrix_elements.

        Params:
        path_to_diag_matrix_elements - path to the file with diagonal eigenvalues
        """

        matrix_elements_raw = np.loadtxt(path_to_diag_matrix_elements)
        matrix_elements_raw = matrix_elements_raw[
            :, -2:
        ]  # take only the right eigenvector

        matrix_elements_re = matrix_elements_raw[:, 0]  # real part
        matrix_elements_im = matrix_elements_raw[:, 1]  # imaginary part

        self.diag_matrix_elements = matrix_elements_re + 1j * matrix_elements_im

    def assert_diag_initialization(self):
        """
        Assertion of the diagonal data initialization.
        """
        assert (
            self.__diag_initialized
        ), f"Diagonal matrix elements and eigenvalues are not initialized for {self.name}!"

    def initialize_hole(
        self,
        path_to_pcur_all,
        hole_kappa,
        n_qn,
        binding_energy,
        g_omega_IR,
        path_to_amp_all=None,
        path_to_phaseF_all=None,
        path_to_phaseG_all=None,
        should_reinitialize=False,
    ):
        """
        Initializes a hole and adds corresponding ionization channels.

        Params:
        path_to_pcur_all - path to file with probabilty current for one photon
        hole_kappa - kappa value of the hole
        n_qn - principal quantum number of the hole
        binding_energy - binding energy of the hole
        g_omega_IR - energy of IR photon used in Fortaran simulations (in Hartree units)
        path_to_amp_all - path to file with amplitudes for one photon
        path_to_phaseF_all - path to file with the phase for larger relativistic component
        of the wave function
        path_to_phaseG_all - path to file with the phase for smaller relativistic component
        of the wave function
        should_reinitialize - tells whether we should reinitialize if the hole was previously
        initialized
        """
        is_initialized = self.is_initialized(n_qn, hole_kappa)

        if not is_initialized or should_reinitialize:
            if is_initialized and should_reinitialize:
                print(
                    f"Reinitialize {self.channels[(n_qn, hole_kappa)].hole.name} hole in {self.name}!"
                )
            self._add_hole_and_channels(
                path_to_pcur_all,
                hole_kappa,
                n_qn,
                binding_energy,
                g_omega_IR,
                path_to_amp_all=path_to_amp_all,
                path_to_phaseF_all=path_to_phaseF_all,
                path_to_phaseG_all=path_to_phaseG_all,
            )

    def is_initialized(self, n_qn, hole_kappa):
        """
        Checks if the hole is initialized (contained in self.channels)

        Params:
        n_qn - principal quantum number of the hole
        hole_kappa - kappa value of the hole

        Returns:
        True if initialized, False otherwise.
        """

        return (n_qn, hole_kappa) in self.channels

    def assert_hole_initialization(self, n_qn, hole_kappa):
        """
        Assertion of the hole initialization.

        Params:
        n_qn - principal quantum number of the hole
        hole_kappa - kappa value of the hole
        """

        assert self.is_initialized(
            n_qn, hole_kappa
        ), f"The hole {construct_hole_name(n_qn, hole_kappa)} in {self.name} is not initialized!"

    def _add_hole_and_channels(
        self,
        path_to_pcur_all,
        hole_kappa,
        n_qn,
        binding_energy,
        g_omega_IR,
        path_to_amp_all=None,
        path_to_phaseF_all=None,
        path_to_phaseG_all=None,
    ):
        """
        Adds ionization channels for the given hole.

        Params:
        path_to_pcur_all - path to file with probabilty current for one photon
        hole_kappa - kappa value of the hole
        n_qn - principal quantum number of the hole
        binding_energy - binding energy of the hole
        g_omega_IR - energy of IR photon used in Fortaran simulations (in Hartree units)
        path_to_amp_all - path to file with amplitudes for one photon
        path_to_phaseF_all - path to file with the phase for larger relativistic component
        of the wave function
        path_to_phaseG_all - path to file with the phase for smaller relativistic component
        of the wave function
        """

        # If the paths to the amplitude and phase files were not specified we assume
        # that they are in the same directory as the pcur file.
        pert_path = (
            os.path.sep.join(path_to_pcur_all.split(os.path.sep)[:-1]) + os.path.sep
        )
        if path_to_amp_all is None:
            path_to_amp_all = pert_path + "amp_all.dat"
        if path_to_phaseF_all is None:
            path_to_phaseF_all = pert_path + "phaseF_all.dat"
        if path_to_phaseG_all is None:
            path_to_phaseG_all = pert_path + "phaseG_all.dat"

        self.channels[(n_qn, hole_kappa)] = Channels(
            path_to_pcur_all,
            path_to_amp_all,
            path_to_phaseF_all,
            path_to_phaseG_all,
            hole_kappa,
            n_qn,
            binding_energy,
            g_omega_IR,
        )
        self.num_channels += 1

    def assert_final_kappa(self, n_qn, hole_kappa, final_kappa):
        """
        Assertion of the final state. Checks if the given final state
        is within possible ionization channels for the given hole.

        Params:
        n_qn - principal quantum number of the initial hole
        hole_kappa - kappa value of the hole
        final_kappa - kappa value of the final state
        """

        assert self.check_final_kappa(
            n_qn, hole_kappa, final_kappa
        ), f"The final state with kappa {final_kappa} is not within channels for {self.channels[(n_qn, hole_kappa)].hole.name} hole in {self.name}!"

    def check_final_kappa(self, n_qn, hole_kappa, final_kappa):
        """
        Checks if the given final state is within ionization channels of the given initial hole.

        Params:
        n_qn - principal quantum number of the initial hole
        hole_kappa - kappa value of the initial hole
        final_kappa - kappa value of the final state

        Returns:
        True if the final state is within ionization channels, False otherwise.
        """

        self.assert_hole_initialization(n_qn, hole_kappa)

        channel = self.channels[(n_qn, hole_kappa)]

        return final_kappa in channel.final_states

    def get_channel_labels_for_hole(self, n_qn, hole_kappa):
        """
        Constructs labels for all ionization channels of the given hole.

        Params:
        n_qn - principal quantum number of the hole
        hole_kappa - kappa value of the hole

        Returns:
        channel_labels - list with labels of all ionization channels
        """

        self.assert_hole_initialization(n_qn, hole_kappa)

        channel_labels = []
        channel = self.channels[(n_qn, hole_kappa)]
        hole_name = channel.hole.name
        for final_state_key in channel.final_states.keys():
            final_state = channel.final_states[final_state_key]
            channel_labels.append(hole_name + " to " + final_state.name)

        return channel_labels
