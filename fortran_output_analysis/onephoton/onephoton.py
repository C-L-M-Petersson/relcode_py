import os
import numpy as np

from fortran_output_analysis.common_utility import (
    l_from_kappa,
    j_from_kappa,
    j_from_kappa_int,
    IonHole,
    load_raw_data,
    l_to_str,
)


class FinalState:
    def __init__(self, kappa, pcur_col_idx):
        self.kappa = kappa
        self.l = l_from_kappa(kappa)
        self.j = j_from_kappa(kappa)
        self.name = l_to_str(self.l) + ("_{%i/2}" % (j_from_kappa_int(kappa)))
        self.pcur_column_index = pcur_col_idx


class Channels:
    def __init__(
        self,
        path_to_pcur,
        path_to_amp_all,
        path_to_phaseF_all,
        path_to_phaseG_all,
        hole_kappa,
        n_qn,
        binding_energy,
    ):
        self.path_to_pcur = path_to_pcur
        self.hole = IonHole(hole_kappa, n_qn, binding_energy)
        self.final_states = {}
        self.raw_data = load_raw_data(path_to_pcur)
        self.raw_amp_data = load_raw_data(path_to_amp_all)
        self.raw_phaseF_data = load_raw_data(path_to_phaseF_all)
        self.raw_phaseG_data = load_raw_data(path_to_phaseG_all)
        self.add_final_states()

    def add_final_states(self):
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
        state = self.final_states[final_kappa]
        column_index = state.pcur_column_index
        rate = self.raw_data[:, column_index]
        return rate

    @staticmethod
    def final_kappas(hole_kappa, only_reachable=True):
        """Returns the possible final kappas that can be reached
        with one photon from an initial state with the given kappa.
        If only_reachable is False, this function will always return
        a list of three elements, even if one of them is 0."""
        mag = np.abs(hole_kappa)
        sig = np.sign(hole_kappa)

        kappas = [sig * (mag - 1), -sig * mag, sig * (mag + 1)]

        if only_reachable:
            # Filter out any occurence of final kappa = 0
            kappas = [kappa for kappa in kappas if kappa != 0]

        return kappas


class OnePhoton:
    """
    Initializes holes for one photon case and reads raw data from the Fortran output
    """

    def __init__(self, atom_name):
        self.name = atom_name
        self.channels = {}
        self.num_channels = 0

    def initialize_hole(
        self,
        path_to_pcur_all,
        hole_kappa,
        n_qn,
        binding_energy,
        path_to_amp_all=None,
        path_to_phaseF_all=None,
        path_to_phaseG_all=None,
    ):
        is_initialized = self.is_initialized(hole_kappa)
        should_reinitialize = True
        if is_initialized:
            should_reinitialize = self.__should_reinitialize(hole_kappa)

        if not is_initialized or should_reinitialize:
            self._add_hole_and_channels(
                path_to_pcur_all,
                hole_kappa,
                n_qn,
                binding_energy,
                path_to_amp_all=path_to_amp_all,
                path_to_phaseF_all=path_to_phaseF_all,
                path_to_phaseG_all=path_to_phaseG_all,
            )

    def is_initialized(self, hole_kappa):

        return hole_kappa in self.channels

    def __assert_hole_initialization(self, hole_kappa):

        assert self.is_initialized(
            hole_kappa
        ), f"The hole with kappa {hole_kappa} is not initialized!"

    def __should_reinitialize(self, hole_kappa):
        """Asks if the hole sould be reinitialized"""

        print(f"The hole with kappa {hole_kappa} is already initialized.")
        answer = input("Do you want to reinitialize? (Type: Yes/No):")
        answer = answer.strip().lower()

        retrial_number = 0
        while (answer != "yes") and (answer != "no"):
            if retrial_number == 3:
                raise RuntimeError("The maximum number of retrials exceeded!")
            answer = input("Invalid answer, type Yes or No:")
            answer = answer.strip().lower()
            retrial_number += 1

        return True if answer == "yes" else False

    def _add_hole_and_channels(
        self,
        path_to_pcur_all,
        hole_kappa,
        n_qn,
        binding_energy,
        path_to_amp_all=None,
        path_to_phaseF_all=None,
        path_to_phaseG_all=None,
    ):
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

        self.channels[hole_kappa] = Channels(
            path_to_pcur_all,
            path_to_amp_all,
            path_to_phaseF_all,
            path_to_phaseG_all,
            hole_kappa,
            n_qn,
            binding_energy,
        )
        self.num_channels += 1

    def check_final_kappa(self, hole_kappa, final_kappa):
        """Checks if given final kappa is within channels for the given hole kappa"""
        self.__assert_hole_initialization(hole_kappa)

        channel = self.channels[hole_kappa]

        return final_kappa in channel.final_states

    def get_channel_labels_for_hole(self, hole_kappa):

        channel_labels = []
        channel = self.channels[hole_kappa]
        hole_name = channel.hole.name
        for final_state_key in channel.final_states.keys():
            final_state = channel.final_states[final_state_key]
            channel_labels.append(hole_name + " to " + final_state.name)

        return channel_labels
