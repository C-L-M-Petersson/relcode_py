import numpy as np
from fortran_output_analysis.constants_and_parameters import g_eV_per_Hartree
from fortran_output_analysis.common_utility import l_from_kappa, l_to_str, \
    wigner_eckart_phase, wigner3j_numerical2, j_from_kappa, \
    j_from_kappa_int, IonHole, load_raw_data, convert_rate_to_cross_section, exported_mathematica_tensor_to_python_list
import os

# ==================================================================================================
#
# ==================================================================================================
class FinalState:
    def __init__(self, kappa, pcur_col_idx):
        self.kappa = kappa
        self.l = l_from_kappa(kappa)
        self.j = j_from_kappa(kappa)
        self.name = l_to_str(self.l) + ("_{%i/2}" % (j_from_kappa_int(kappa)))
        self.pcur_column_index = pcur_col_idx


# ==================================================================================================
#
# ==================================================================================================
class OnePhoton:
    def __init__(self, atom_name):
        self.name = atom_name
        self.channels = {}
        self.num_channels = 0

    def add_channels_for_hole(self, path_to_pcur_all, hole_kappa, n_qn, path_to_amp_all = None, path_to_phaseF_all = None, path_to_phaseG_all = None):
        #If the paths to the amplitude and phase files were not specified we assume
        #that they are in the same directory as the pcur file.
        pert_path = os.path.sep.join(path_to_pcur_all.split(os.path.sep)[:-1]) + os.path.sep
        if path_to_amp_all is None:
            path_to_amp_all = pert_path + "amp_all.dat"
        if path_to_phaseF_all is None:
            path_to_phaseF_all = pert_path + "phaseF_all.dat"
        if path_to_phaseG_all is None:
            path_to_phaseG_all = pert_path + "phaseG_all.dat"

        self.channels[hole_kappa] = Channels(path_to_pcur_all, path_to_amp_all, path_to_phaseF_all, path_to_phaseG_all, hole_kappa, n_qn)
        self.num_channels += 1

    def get_channel_labels_for_hole(self, hole_kappa):
        channel_labels = []
        channel = self.channels[hole_kappa]
        hole_name = channel.hole.name
        for final_state_key in channel.final_states.keys():
            final_state = channel.final_states[final_state_key]
            channel_labels.append(hole_name + " to " + final_state.name)

        return channel_labels

    def get_omega_eV(self):
        assert (len(self.channels) > 0)
        first_kappa = list(self.channels.keys())[0]
        first_channel = self.channels[first_kappa]
        omega_eV = first_channel.raw_data[:, 0] * g_eV_per_Hartree  # omega energies in atomic units in file.
        return omega_eV

    def get_omega_Hartree(self):
        assert (len(self.channels) > 0)
        first_kappa = list(self.channels.keys())[0]
        first_channel = self.channels[first_kappa]
        omega_eV = first_channel.raw_data[:, 0] # omega energies in atomic units in file.
        return omega_eV

    def get_partial_cross_section(self, hole_kappa, final_kappa, divide_omega=True):
        # Depending on conventions when creating the dipole elements in the Fortran program we might
        # have to divide or multiply by the photon energy (omega) when calculating cross sections.
        # Usually it is correct to divide by omega, and that is default behaviour of this function.
        hole = self.channels[hole_kappa]
        rate = hole.get_rate_for_channel(final_kappa)
        omega = self.get_omega_Hartree()
        cross_section = convert_rate_to_cross_section(rate, omega, divide_omega)
        return cross_section

    def get_total_cross_section(self, divide_omega=True):
        # Use omega to initialise total cs array:
        omega = self.get_omega_Hartree()
        N = len(omega)
        total_cs = np.zeros(N)
        for channel_key in self.channels.keys():
            channel = self.channels[channel_key]
            hole_kappa = channel.hole.kappa
            for final_key in channel.final_states.keys():
                final_state = channel.final_states[final_key]
                final_kappa = final_state.kappa
                cs = self.get_partial_cross_section(hole_kappa, final_kappa, divide_omega)
                total_cs += cs

        return total_cs

    def get_matrix_element_with_phase_for_channel(self, hole_kappa, final_kappa):
        """Returns the value of the matrix element after one photon as amp*[e^(i*phase_of_F), e^(i*phase_of_G)]."""
        channel = self.channels[hole_kappa]
        final_state = channel.final_states[final_kappa]
        #We assume that the data is sorted the same in amp_all and phaseF_all as in pcur_all
        #this is true at time of writing (2022-05-23).
        column_index = final_state.pcur_column_index
        return channel.raw_amp_data[:,column_index]*[np.exp(1j*channel.raw_phaseF_data[:,column_index]), np.exp(1j*channel.raw_phaseG_data[:,column_index])]


# ==================================================================================================
#
# ==================================================================================================

def final_kappas(hole_kappa, only_reachable=True):
    """Returns the possible final kappas that can be reached
    with one photon from an initial state with the given kappa.
    If only_reachable is False, this function will always return
    a list of three elements, even if one of them is 0."""
    mag = np.abs(hole_kappa)
    sig = np.sign(hole_kappa)

    kappas = [sig*(mag-1), -sig*mag, sig*(mag+1)]

    if only_reachable:
        #Filter out any occurence of final kappa = 0
        kappas = [kappa for kappa in kappas if kappa != 0]

    return kappas


class Channels:
    def __init__(self, path_to_pcur, path_to_amp_all, path_to_phaseF_all, path_to_phaseG_all, hole_kappa, n_qn):
        self.path_to_pcur = path_to_pcur
        self.hole = IonHole(hole_kappa, n_qn)
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

        #This code should reproduce the previous implementation
        possible_final_kappas = final_kappas(kappa_hole, only_reachable=False)

        # This is for getting the data from the pcur files. The first column is the photon energy.
        pcur_column_index = 1
        for kappa in possible_final_kappas:
            if (kappa != 0):
                self.final_states[kappa] = FinalState(kappa, pcur_column_index)

            pcur_column_index += 1

    def get_rate_for_channel(self, final_kappa):
        state = self.final_states[final_kappa]
        column_index = state.pcur_column_index
        rate = self.raw_data[:, column_index]
        return rate


# ==================================================================================================
#
# ==================================================================================================
class ChannelsOld:
    def __init__(self, atom_name):
        self.name = atom_name
        self.ion_holes = {}
        self.eV_per_Hartree = 27.211396641307999
        self.num_holes = 0

    def add_ion_hole(self, path, kappa, n_qn):
        self.ion_holes[kappa] = IonHole(path, kappa, n_qn)
        self.num_holes += 1

    def get_omega_eV(self):
        assert (len(self.ion_holes) > 0)
        first_kappa = list(self.ion_holes.keys())[0]
        first_hole = self.ion_holes[first_kappa]
        omega_eV = first_hole.raw_data[:, 0] * self.eV_per_Hartree  # omega energies in atomic units in file.
        return omega_eV

    def print_channels(self):
        for hole_kappa in self.ion_holes:
            hole = self.ion_holes[hole_kappa]
            for final_state_kappa in hole.final_states:
                final_state = hole.final_states[final_state_kappa]
                channel_name = hole.name + " -> " + final_state.name
                print(channel_name)

    def get_channel_label_lj(self, l_hole, j_hole, l_final, j_final):
        hole_kappa = kappa_from_l_and_j(l_hole, j_hole)
        final_kappa = kappa_from_l_and_j(l_final, j_final)
        return get_channel_label(hole_kappa, final_kappa)

    def get_channel_label_kappa(self, hole_kappa, final_kappa):
        hole = self.ion_holes[hole_kappa]
        final_state = hole.final_states[final_kappa]
        return hole.name + " \\rightarrow " + final_state.name

    def get_channel_label_for_filename(self, hole_kappa, final_kappa):
        hole = self.ion_holes[hole_kappa]
        final_state = hole.final_states[final_kappa]
        l_hole = hole.l
        n_hole = hole.n
        l_final = final_state.l
        hole_str = str(n_hole) + l_to_str(l_hole) + ("%i" % (j_from_kappa_int(hole_kappa)))
        final_str = l_to_str(l_final) + ("%i" % (j_from_kappa_int(final_kappa)))
        return hole_str + "_to_" + final_str

    def get_cross_section_for_channel(self, hole_kappa, final_kappa):
        hole = self.ion_holes[hole_kappa]
        final_state = hole.final_states[final_kappa]
        omega_au = hole.raw_data[:, 0]
        rate = hole.raw_data[:, final_state.pcur_column_index]
        cs = convert_rate_to_cross_section(rate, omega_au, divide=True)
        return cs

    def compute_wigner3j_for_channel(self, hole_kappa, final_kappa, mj):
        mjj = int(2 * mj)
        hole = self.ion_holes[hole_kappa]
        final = hole.final_states[final_kappa]
        j_hole = j_from_kappa_int(hole_kappa)
        j_final = j_from_kappa_int(final_kappa)
        if (mjj > j_hole or mjj > j_final):
            return
        # print("j_hole, j_final, mj: %i/2 %i/2 %i/2" % (j_hole, j_final, mjj))
        K = 1
        q = 0
        w3j = wigner_3j(j_final / 2, K, j_hole / 2, -mjj / 2, q, mjj / 2)
        # print(w3j, sympy_to_num(w3j))
        return sympy_to_num(w3j)

    def get_wigner_eckart_phase(self, final_kappa, mj):
        return np.power(-1.0, (j_from_kappa(final_kappa) - mj))

    def get_total_cross_section(self):
        assert (self.num_holes > 0)
        omega = self.get_omega_eV()
        tot_cs = np.zeros(len(omega))
        for hole_kappa in self.ion_holes.keys():
            hole = self.ion_holes[hole_kappa]
            for final_kappa in hole.final_states.keys():
                part_cs = self.get_cross_section_for_channel(hole_kappa, final_kappa)
                tot_cs += part_cs

        return tot_cs

def get_integrated_one_photon_cross_section(hole_kappa, M1, M2, abs_emi_or_cross, path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "formula_coefficients","one_photon","integrated_intensity"), threshold=1e-10):
    """This function returns the value of the integrated cross section for a photoelectron that has absorbed one photon"""

    if abs_emi_or_cross != "abs" and abs_emi_or_cross != "emi" and abs_emi_or_cross != "cross":
        raise ValueError(f"abs_emi_or_cross can only be 'abs', 'emi', or 'cross', not {abs_emi_or_cross}")

    if len(M1[0]) != len(M2[0]):
        raise ValueError("the length of the input matrix elements must be the same")

    length = len(M1[0])

    if path[-1] is not os.path.sep:
        path = path + os.path.sep

    try:
        with open(path + f"integrated_intensity_{hole_kappa}.txt","r") as coeffs_file:
            coeffs_file_contents = coeffs_file.readlines()
    except OSError as e:
        raise NotImplementedError("the given initial kappa is not yet implemented, or the file containing the coefficients could not be found")

    coeffs = exported_mathematica_tensor_to_python_list(coeffs_file_contents[2])

    integrated_cross_section = np.zeros(length, dtype="complex128")
    for i in range(3):
        integrated_cross_section += coeffs[i]*M1[i]*np.conj(M2[i])

    #for i in range(3):
    #    np.savetxt(f"m_elem_1ph_gt_kappa_{i}.txt", M1[i])
    #    np.savetxt(f"m_elem_1ph_lt_kappa_{i}.txt", M2[i])
    #exit()

    if abs_emi_or_cross == "cross":
        abs_emi_or_cross = "complex"
    else:
        #If we are looking at the diagonal terms the results are real
        values = integrated_cross_section[~np.isnan(integrated_cross_section)] #Filter out the nans first, as they mess up boolean expressions (nan is not itself).
        assert all(np.abs(np.imag(values)) < threshold), "The integrated cross section had a non-zero imaginary part when it shouldn't. Check the input matrix elements or change the threshold for the allowed size of the imaginary part"
        integrated_cross_section = np.real(integrated_cross_section)

    label = f"$I_0^{{{abs_emi_or_cross}}}$ from $\\kappa_0=${hole_kappa}"

    return integrated_cross_section, label


def get_one_photon_asymmetry_parameter(hole_kappa, M1, M2, abs_emi_or_cross, path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "formula_coefficients", "one_photon", "asymmetry_coeffs"), threshold=1e-10):
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
    If the asymmetry parameter has an imaginary part larger than the input threshold when half_of_cross_term == False,
    this will trigger an assertion error. This threshold can be modified with the threshold input"""

    if abs_emi_or_cross != "abs" and abs_emi_or_cross != "emi" and abs_emi_or_cross != "cross":
        raise ValueError(f"abs_emi_or_cross can only be 'abs', 'emi', or 'cross' not {abs_emi_or_cross}")

    if path[-1] is not os.path.sep:
        path = path + os.path.sep

    data_size = 0
    if len(M1[0]) != len(M2[0]):
        raise ValueError(f"the length of the two matrix elements must be the same, but they are {len(M1[0])} and {len(M2[0])}")
    else:
        data_size = len(M1[0])

    #Try opening the needed file.
    try:
        with open(path + f"asymmetry_coeffs_2_{hole_kappa}.txt","r") as coeffs_file:
            coeffs_file_contents = coeffs_file.readlines()
    except OSError as e:
        print(e)
        raise NotImplementedError("the formula for that initial kappa is not yet implemented, or the file containing the coefficients could not be found")

    #Read in the coefficients in front of all the different combinations of matrix elements in the numerator.
    numerator_coeffs = np.array(exported_mathematica_tensor_to_python_list(coeffs_file_contents[3]))

    #Read in the coefficients in front of the absolute values in the denominator.
    denominator_coeffs = exported_mathematica_tensor_to_python_list(coeffs_file_contents[4])

    numerator = np.zeros(data_size, dtype="complex128")
    denominator = np.zeros(data_size, dtype="complex128")
    for i in range(3):
        denominator += denominator_coeffs[i]*M1[i]*np.conj(M2[i])#np.abs(M[i])**2
        #for j in range(i, 3):
        for j in range(3):
            #if not half_of_cross_terms:
            #    numerator += numerator_coeffs[i,j]*2*np.real(M1[i]*np.conj(M2[j]))
            #else:
            #    numerator += numerator_coeffs[i,j]*M1[i]*np.conj(M2[j])
            numerator += numerator_coeffs[i,j]*M1[i]*np.conj(M2[j])

    parameter = numerator/denominator

    if abs_emi_or_cross != "cross":
        # When looking at the asymmetry parameter from the diagonal part
        # or the full cross part, the result is a real number
        values = parameter[~np.isnan(parameter)] #Filter out the nans first, as they mess up boolean expressions (nan is not itself).
        assert all(np.abs(np.imag(values)) < threshold), "The asymmetry parameter had a non-zero imaginary part when it shouldn't. Check the input matrix elements or change the threshold for the allowed size of the imaginary part"
        parameter = np.real(parameter)

    if abs_emi_or_cross == "cross":
        abs_emi_or_cross = "complex"

    label = f"$\\beta_2^{{{abs_emi_or_cross}}}$"

    return parameter, label
