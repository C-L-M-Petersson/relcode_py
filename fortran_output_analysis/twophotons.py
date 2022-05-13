import numpy as np
import re  # Regular expressions
from itertools import islice  # Slicing when reading lines from Fortran files.
from fortran_output_analysis.constants_and_parameters import g_eV_per_Hartree, g_omega_IR
from fortran_output_analysis.common_utility import l_from_kappa, l_to_str, \
     wigner_eckart_phase, wigner3j_numerical2, j_from_kappa, \
     j_from_kappa_int, IonHole, phase, exported_mathematica_tensor_to_python_list, mag, cross, coulomb_phase
from sympy.physics.wigner import wigner_3j, wigner_6j
from input_to_fortran.parse_user_input_file import parse_user_input_file
import os

# ==================================================================================================
#
# ==================================================================================================
class IonisationPath:
    def __init__(self, intermediate_kappa, final_kappa, file_col_idx, col_idx):
        self.kappa_intermediate = intermediate_kappa
        self.l_intermediate = l_from_kappa(intermediate_kappa)
        self.j_intermediate = j_from_kappa(intermediate_kappa)
        self.kappa_final = final_kappa
        self.l_final = l_from_kappa(final_kappa)
        self.j_final = j_from_kappa(final_kappa)
        self.name_intermediate = "" + l_to_str(self.l_intermediate) + ("_{%i/2}" % (j_from_kappa_int(intermediate_kappa)))
        self.name_final = "" + l_to_str(self.l_final) + ("_{%i/2}" % (j_from_kappa_int(final_kappa)))

        # We don't store zero columns from the input file,
        # so the index in the file and the index used to retrieve data will differ.
        self.file_column_index = file_col_idx
        self.column_index = col_idx

    def full_name(self):
        return self.name_intermediate + " to " + self.name_final


# ==================================================================================================
#
# ==================================================================================================
class MatrixElements:
    # This class handles the fortran output data
    # for each channel described by a two-photon (XUV+IR) interaction:
    # hole - intermediate - final
    # It also handles the summation over intermediate states required
    # for a "measurable" description of the channel.
    def __init__(self, path, hole_kappa, hole_n, abs_or_emi):
        self.is_initialised = False
        self.path = path
        self.hole = IonHole(hole_kappa, hole_n)
        self.ionisation_paths = {}

        with open(path,"r") as file:

            # Sets up what the available ionisation paths are, and what columns in the file it is corresponding to.
            parse_first_line_from_fortran_matrix_element_output_file(file, self.hole, self.ionisation_paths)
            self.number_of_ionisation_paths = len(self.ionisation_paths)

            # We use this list to only pick the non-zero data points after parsing raw data below.
            self.path_col_indices = []
            for ionisation_path in self.ionisation_paths.values():
                self.path_col_indices.append(ionisation_path.file_column_index)

            # These will be filled with parsing the rest of the file.
            self.raw_data_real = np.zeros(1)
            self.raw_data_imag = np.zeros(1)
            # For any energy the matrix element is printed at each breakpoint from the Fortran program.
            # So here we can control what breakpoints we choose.
            self.breakpoint_step = 5

            self.raw_data_real, self.raw_data_imag = \
                parse_matrix_element_raw_data_from_fortran_output_file(file,
                                                                       self.number_of_ionisation_paths,
                                                                       self.path_col_indices,
                                                                       self.breakpoint_step)

        # Errors for this are catched outside this class.
        abs_or_emi_string = "emission"
        if(abs_or_emi == "abs"):
            abs_or_emi_string = "absorption"

        self.name = self.hole.name + " " + abs_or_emi_string + " matrix elements."

        print("Added:")
        print(self.name)
        self.is_initialised = True

    def get_ionisation_path(self, intermediate_kappa, final_kappa):
        if self.is_initialised:
            key_tuple = (self.hole.kappa, intermediate_kappa, final_kappa)
            ionisation_path = self.ionisation_paths[key_tuple]
            column_index = ionisation_path.column_index
            z = self.raw_data_real[:, column_index] + 1j * self.raw_data_imag[:, column_index]
            name = self.hole.name + "$ to $" + ionisation_path.full_name()
            return z, name
        else:
            raise ValueError("MatrixElements not initialised!")

    def get_ionisation_path_summed_over_intermediate(self, final_kappa, mj):

        paths_summed_over_intermediate_states = sum_over_intermediate_states_including_3j_symbols(self.hole.kappa,
                                                                                                  final_kappa,
                                                                                                  self.raw_data_real,
                                                                                                  self.raw_data_imag,
                                                                                                  self.ionisation_paths,
                                                                                                  mj)

        for state_tuple in paths_summed_over_intermediate_states.keys():

            if final_kappa == state_tuple[1]:
                final_label = l_to_str(l_from_kappa(final_kappa)) + ("_{%i/2}" % (j_from_kappa_int(final_kappa)))
                name = self.hole.name + "$  to  $" + final_label
                return paths_summed_over_intermediate_states[state_tuple], name

        # If channel doesn't exist we return None and handle error outside this function.
        return None

    def get_summed_channels(self, mj):
        # This function just gets the hole- and final-kappas present, ie implies the sum over all intermediate.
        # It returns a list of the unique tuples (hole_kappa, final_kappa).
        # It can be used as input to getting all the data for these channels summed over intermediate states.
        hole_kappa = self.hole.kappa
        channels = []
        for path_tuple in self.ionisation_paths.keys():
            final_kappa = path_tuple[2]

            j_hole = j_from_kappa_int(hole_kappa)
            j_final = j_from_kappa_int(final_kappa)
            mjj = int(2 * mj)
            if (np.abs(mjj) > j_hole or np.abs(mjj) > j_final):
                continue

            if (hole_kappa, final_kappa) not in channels:
                channels.append((hole_kappa, final_kappa))

        return channels


# ==================================================================================================
#
# ==================================================================================================
class TwoPhotons:
    def __init__(self, atom_name):
        self.atom_name = atom_name
        self.matrix_elements_abs = {}
        self.matrix_elements_emi = {}
        self.eV_per_Hartree = g_eV_per_Hartree
        self.omega_path = ""
        self.omega_Hartree = 0
        self.omega_eV = 0

    def add_omega(self, path):
        self.omega_path = path
        self.omega_Hartree = np.loadtxt(path)
        self.omega_eV = self.omega_Hartree * self.eV_per_Hartree

    # Add matrix elements after absorption or emission of an IR photon
    # for a particular hole (from XUV absorption),
    # using the output data from the Fortran program.
    def add_matrix_elements(self, path, abs_or_emi, hole_kappa, hole_n):

        if (abs_or_emi == "abs"):
            if (path.find("abs") == -1):
                raise ValueError("Specified path not compliant with abs/emi convention! Got: path.find('abs') == -1")
            else:
                self.matrix_elements_abs[hole_kappa] = MatrixElements(path, hole_kappa, hole_n, abs_or_emi)

        elif (abs_or_emi == "emi"):
            if (path.find("emi") == -1):
                raise ValueError("Specified path not compliant with abs/emi convention! Got: path.find('emi') == -1")
            else:
                self.matrix_elements_emi[hole_kappa] = MatrixElements(path, hole_kappa, hole_n, abs_or_emi)

        else:
            raise ValueError("Need to specify emission ('emi') or absorption ('abs') when adding matrix elements data!")

        return

    # This is a method to get the data that is summed over intermediate states and over m values.
    def get_matrix_elements_for_hole_to_final(self, hole_kappa, final_kappa, abs_or_emi, mj):

        retval = np.zeros(1)
        if (abs_or_emi == "abs"):
            retval, name = \
                self.matrix_elements_abs[hole_kappa].get_ionisation_path_summed_over_intermediate(final_kappa, mj)

        elif (abs_or_emi == "emi"):
            retval, name = \
                self.matrix_elements_emi[hole_kappa].get_ionisation_path_summed_over_intermediate(final_kappa, mj)
        else:
            raise ValueError("Need to specify emission ('emi') or absorption ('abs') when getting matrix element data!")

        if len(retval) < 2:
            raise ValueError("Couldn't get matrix elements for hole_kappa %i and final_kappa %i, \n"
                             "get_ionisation_path_summed_over_intermediate_and_mj returned None." % (hole_kappa, final_kappa))
        else:
            return retval, name

    def get_hole_to_final_channels(self, hole_kappa, abs_or_emi, mj):
        channels = []
        if (abs_or_emi == "abs"):
            channels = self.matrix_elements_abs[hole_kappa].get_summed_channels(mj)

        elif (abs_or_emi == "emi"):
            channels = self.matrix_elements_emi[hole_kappa].get_summed_channels(mj)
        else:
            raise ValueError("Need to specify emission ('emi') or absorption ('abs') when getting matrix element data!")

        return channels

    def get_matrix_element_for_intermediate_resolved_channel(self, hole_kappa, intermediate_kappa, final_kappa, abs_or_emi):
        retval = np.zeros(1)
        if (abs_or_emi == "abs"):
            retval, name = \
                self.matrix_elements_abs[hole_kappa].get_ionisation_path(intermediate_kappa, final_kappa)

        elif (abs_or_emi == "emi"):
            retval, name = \
                self.matrix_elements_emi[hole_kappa].get_ionisation_path(intermediate_kappa, final_kappa)
        else:
            raise ValueError("Need to specify emission ('emi') or absorption ('abs') when getting matrix element data!")

        if len(retval) < 2:
            raise ValueError("Couldn't get matrix elements for hole_kappa %i and final_kappa %i, \n"
                             "get_ionisation_path_summed_over_intermediate_and_mj returned None." % (
                             hole_kappa, final_kappa))
        else:
            return retval, name
    

    def get_coupled_matrix_element(self, hole_kappa, abs_or_emi, final_kappa):
        """Computes the value of the specified coupled matrix element.
        This function also adds in the non-matrix element phases from the fortran program since that is
        hard to do after coupling."""
        if abs_or_emi == "abs":
            fortranM = self.matrix_elements_abs[hole_kappa]
        elif abs_or_emi == "emi":
            fortranM = self.matrix_elements_emi[hole_kappa]
        else:
            raise ValueError("Need to specify emission ('emi') or absorption ('abs') when getting matrix element data!")

        #Get the length of the array of data points by looking up the index of the first open channel and checking
        #the row length at that column.
        random_col = fortranM.ionisation_paths[next(iter(fortranM.ionisation_paths.keys()))].column_index
        energy_size = len(fortranM.raw_data_real[:,random_col])
        coupled_matrix_element = np.zeros(energy_size,dtype='complex128')

        #Get the data from the correct phase file
        phase_data = self._raw_short_range_phase(abs_or_emi, hole_kappa)

        for K in [0,2]:
            #Loop through all the ionisation paths
            for (loop_hole_kappa, intermediate_kappa,loop_final_kappa) in fortranM.ionisation_paths.keys():
                #Only use those that match the requested initial and final state
                if loop_hole_kappa == hole_kappa and loop_final_kappa == final_kappa:
                    # Get j values
                    hole_j = j_from_kappa(hole_kappa)
                    intermediate_j = j_from_kappa(intermediate_kappa)
                    final_j = j_from_kappa(final_kappa)

                    # Get the column index of the raw fortran output file that contains the requested ionisation path
                    col_index = fortranM.ionisation_paths[(hole_kappa,intermediate_kappa,final_kappa)].column_index

                    #Add in the short range phase from the fortran program
                    matrix_element = fortranM.raw_data_real[:,col_index] + 1j*fortranM.raw_data_imag[:,col_index]
                    matrix_element *= np.exp(1j*phase_data[:,col_index])
                    coupled_matrix_element += phase(hole_j + final_j + K)*(2*K+1)*float(wigner_3j(1,1,K,0,0,0))*matrix_element*float(wigner_6j(1,1,K,hole_j,final_j,intermediate_j))
        
        return coupled_matrix_element

    def _raw_short_range_phase(self, abs_or_emi, hole_kappa):
        """Returns the short range phase for the given channel.
        The channels are organized in the same way for the return value from this function
        As the raw_data_real and raw_data_imag items of a MatrixElement object"""

        if abs_or_emi == "abs":
            M = self.matrix_elements_abs[hole_kappa]
        elif abs_or_emi == "emi":
            M = self.matrix_elements_emi[hole_kappa]
        else:
            raise ValueError(f"abs_or_emi can only be 'abs' or 'emi' not {abs_or_emi}")

        #Determine the name of the data file
        phase_file_name = "phase_" + abs_or_emi + f"_{hole_kappa}_{M.hole.n - l_from_kappa(hole_kappa)}.dat"
        
        #Remove the string after the last "/" in the phase that points to the matrix element file
        #and replace it with the name of the phase data file
        phase_path = os.path.sep.join(M.path.split(os.path.sep)[:-1]) + os.path.sep + phase_file_name
        
        raw_phase_data = np.loadtxt(phase_path)

        return raw_phase_data


    @staticmethod
    def final_kappas(hole_kappa, only_reachable=True):
        """Returns a list of the kappa quantum numbers that are reachable with
        two photons from the state with the given initial kappa
        If only_reachable is set to true the function will only return kappa
        values that can be reached from the initial kappa, otherwise it will
        always return the five 'theoretically possible' channels"""

        sig = np.sign(hole_kappa)
        mag = np.abs(hole_kappa)

        if mag == 1 and only_reachable:
            return [sig*mag, -sig*(mag+1), sig*(mag+2)]
        elif mag == 2 and only_reachable:
            return [-sig*(mag-1), sig*mag, -sig*(mag+1), sig*(mag+2)]
        else:
            #These are the 'theoretically possible' channels.
            return [sig*(mag-2), -sig*(mag-1), sig*mag, -sig*(mag+1), sig*(mag+2)]


def get_asymmetry_parameter(n, hole_kappa, M1, M2, abs_emi_or_cross, half_of_cross_terms=False, path=os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "asymmetry_coeffs", threshold=1e-10):
    """This function returns the value of the
    n:th asymmetry parameter for a state defined by hole_kappa.
    M1 and M2 contain the matrix elements and other phases of the wave function organized according to their final kappa liek so:
    m = |hole_kappa|
    s = sign(hole_kappa)
    MX = [s(m-2), -s(m-1), sm, -s(m+1), s(m+2)]
    The full signal looks something like S = M_abs M_abs^* + M_emi M_emi^* + M_abs M_emi^* + M_emi M_abs^*
    Each term in this sum contains two matrix elements, these are the inputs labeled M1 and M2 in this function.
    This function computes the contribution to the asymmetry parameter from either the diagonal terms or the corss terms.
    So if you want to compute the asymmetry parameter for the cross terms you would put M1 = M_abs and M2 = M_emi.
    If you want to use only half the cross term (e.g. you want complex parameters for delay calculations) set half_of_cross_terms=True.
    If you want to use some other values for the coefficients used in the calculation than the default,
    set path = "path/to/folder/containing/coefficient/files".
    If the asymmetry parameter has an imaginary part larger than the input threshold when half_of_cross_term == False,
    this will trigger an assertion error. This threshold can be modified with the threshold input"""

    if abs_emi_or_cross != "abs" and abs_emi_or_cross != "emi" and abs_emi_or_cross != "cross":
        raise ValueError(f"abs_emi_or_cross can only be 'abs', 'emi' or 'cross' not {abs_emi_or_cross}")

    if len(M1[0]) != len(M2[0]):
        raise ValueError("the matrix elements contain a different number of points")
    
    #If the path to the coefficient files does not end in a path separator, add it.
    if path[-1] is not os.path.sep:
        path = path + os.path.sep

    #Try opening the needed file.
    try:
        with open(path + f"asymmetry_coeffs_{n}_{hole_kappa}.txt","r") as coeffs_file:
            coeffs_file_contents = coeffs_file.readlines()
    except OSError as e:
        print(e)
        raise NotImplementedError("the given combination of initial kappa and n is not yet implemented, or the file containing the coefficients could not be found")

    #Read in the n and hole_kappa values found in the file.
    read_n, read_hole_kappa = exported_mathematica_tensor_to_python_list(coeffs_file_contents[1])
    #If they do not match the ones given to the function, something has gone wrong.
    if read_n != n or read_hole_kappa != hole_kappa:
        raise ValueError("the n or hole_kappa in the coefficients file was not the same as those given to the function")

    #Read in the coefficients in front of the absolute values in the denominator.
    denominator_coeffs = exported_mathematica_tensor_to_python_list(coeffs_file_contents[3])

    #Read in the coefficients in front of all the different combinations of matrix elements in the numerator.
    numerator_coeffs = np.array(exported_mathematica_tensor_to_python_list(coeffs_file_contents[5]))

    numerator = np.zeros(len(M1[0]), dtype="complex128")
    denominator = np.zeros(len(M1[0]), dtype="complex128")
    for i in range(5):
        #compute the 'integrated cross section' denominator.
        #This is not necessarily the integrated cross section as various numerical
        #factors could have canceled in the Mathematica computation.
        denominator += denominator_coeffs[i]*M1[i]*np.conj(M2[i])

        for j in range(i,5):
            #Multiply each combination of matrix elements with its coefficient.
            if i == j:
                #If it's a diagonal term we multiply the coefficient with the magnitue of the matrix element
                numerator += numerator_coeffs[i,j]*M1[i]*np.conj(M2[i])
            else:
                #otherwise we multiply with the cross term between the two matrix elements
                if not half_of_cross_terms:
                    numerator += numerator_coeffs[i,j]*2*np.real(M1[i]*np.conj(M2[j]))
                else:
                    #unless the caller requested that the conjugate part of the cross term should be ignored
                    numerator += numerator_coeffs[i,j]*M1[i]*np.conj(M2[j])

    parameter = numerator/denominator
    if not half_of_cross_terms:
        # When looking at the asymmetry parameter from the diagonal part
        # or the full cross part, the result is a real number
        values = parameter[~np.isnan(parameter)] #Filter out the nans first, as they mess up boolean expressions (nan is not itself).
        assert all(np.abs(np.imag(values)) < threshold), "The asymmetry parameter had a non-zero imaginary part when it shouldn't. Check the input matrix elements or change the threshold for the allowed size of the imaginary part"
        parameter = np.real(parameter)
        
    if half_of_cross_terms:
        abs_emi_or_cross = "complex"
    label = f"$\\beta_{n}^{{{abs_emi_or_cross}}}$"

    return parameter, label


def parse_first_line_from_fortran_matrix_element_output_file(file, in_hole, ionisation_paths):
    # The first line contains information about what is in each column of the output file.
    # They are the kappas for the hole - intermediate - final channels, according to:
    # <hole> <offset_don't_care> <intermediate1> <intermediate2> <intermediate3> <final1> <final2> ... <final9>
    first_line = file.readline().rstrip()
    # split using regex - The kappas are separated by spaces.
    split_first_line = re.split("\s+", first_line)
    # For some reason we get an extra first element that is an empty string.
    # We discard this
    split_first_line = split_first_line[1:]

    # Catch input error here:
    hole_kappa = int(split_first_line[0])
    assert hole_kappa == in_hole.kappa, \
        "Mismatch between hole kappa read from file, and the input to Channels constructor"

    #print(split_first_line)

    intermediate_kappas_str = split_first_line[2:5]
    #print(intermediate_kappas_str)

    final_kappas_str = split_first_line[5:]
    #print(final_kappas_str)

    final_col_index = 0
    raw_data_col_index = 0
    intermediate_stride = 3  # There are three possible final states (including zero for non-channel) for each intermediate.
    intermediate_index = 0
    for intermediate_kappa_str in intermediate_kappas_str:

        intermediate_kappa = int(intermediate_kappa_str)

        i = intermediate_index
        for j in range(3):  # 3 finals per kappa.
            final_index = j + i * intermediate_stride
            final_kappa = int(final_kappas_str[final_index])
            if (intermediate_kappa != 0 and final_kappa != 0):
                ionisation_paths[(hole_kappa, intermediate_kappa, final_kappa)] = \
                    IonisationPath(intermediate_kappa, final_kappa, final_col_index, raw_data_col_index)
                raw_data_col_index += 1

            final_col_index += 1 # note how this is inceremented even though we have a zero column.

        intermediate_index += 1


def parse_matrix_element_raw_data_from_fortran_output_file(file,
                                                           number_of_ionisation_paths,
                                                           non_zero_col_indices,
                                                           breakpoint_step):
    N = 9  # we know we have 9 columns with real/imag pairs
    real_line_np = np.zeros(N, dtype=np.double)
    imag_line_np = np.zeros(N, dtype=np.double)

    real_dynamic = []
    imag_dynamic = []

    # Read rest of lines with actual matrix elements
    for line in islice(file, 1, None, breakpoint_step):  # Skip first line and get every breakpoint_step:th

        line = line.replace(" ", "")  # remove whitespace
        line = line.split(")(")  # split by parentheses
        line[0] = line[0].replace("(", "")  # remove stray parenthesis from first element.
        line[-1] = line[-1].replace(")\n", "")  # remove crap from last element
        # print(line)

        np_index = 0
        for real_imag_pair in line:
            real, imag = real_imag_pair.split(",")
            real_line_np[np_index] = np.double(real)
            imag_line_np[np_index] = np.double(imag)
            np_index += 1

        # print(real_line_np)

        real_dynamic.append(np.copy(real_line_np))
        imag_dynamic.append(np.copy(imag_line_np))

    raw_data_real = np.array(real_dynamic)
    raw_data_imag = np.array(imag_dynamic)

    return raw_data_real[:, non_zero_col_indices], raw_data_imag[:, non_zero_col_indices]


def sum_over_intermediate_states_including_3j_symbols(hole_kappa, final_kappa,
                                                      raw_real, raw_imag,
                                                      ionisation_paths, mj):
    # This function gives back a dictionary containing the two-photon matrix elements (as a function of energy).
    # They are from an initial state (hole) to the final continuum state, summed over all possible intermediate
    # continuum states. We have to provide the mj value to correctly describe the states. If the mj value is not
    # valid for any of the channels we throw an error.
    # Note that this assumes linearly polarised light, so mj will be the same for initial-intermediate-final states.

    paths_summed = {}
    # First we check what has to be summed over.
    # For each unique final state we check what the intermediates are.
    unique_states = []
    all_j_numbers = []

    # NOTE(anton): This is kind of weird code right now, I did it more generally before but at least this
    # checks if the hole and final kappas are states that actually exist.
    for path_tuple in ionisation_paths.keys():
        if (hole_kappa, final_kappa) not in unique_states:
            unique_states.append((hole_kappa, final_kappa))

        for kappa in path_tuple:
            j = j_from_kappa(kappa)
            all_j_numbers.append(j)

    max_mj = np.max(all_j_numbers)
    #print("max mj: ", max_mj)

    N = raw_real.shape[0]
    summed_real = np.zeros((N, len(unique_states)), dtype=np.double)
    summed_imag = np.zeros((N, len(unique_states)), dtype=np.double)
    # For each unique hole and final state we sum over the intermediate j.

    if np.abs(mj) <= max_mj:
        for path_tuple in ionisation_paths.keys():
            ionisation_path = ionisation_paths[path_tuple]
            col_idx = ionisation_path.column_index

            #hole_kappa = path_tuple[0]
            intermediate_kappa = path_tuple[1]
            #final_kappa = path_tuple[2]

            j_hole = j_from_kappa_int(hole_kappa)
            j_intermediate = j_from_kappa_int(intermediate_kappa)
            j_final = j_from_kappa_int(final_kappa)
            mjj = int(2 * mj)
            if (np.abs(mjj) > j_hole or np.abs(mjj) > j_final or np.abs(mjj) > j_intermediate):
                # This will result in a zero 3j symbol for sure, so skip this.
                continue

            #
            unique_state_index = -1
            for unique_state in unique_states:
                unique_state_index += 1
                if (hole_kappa, final_kappa) == unique_state:
                    break

            # (-1)^(j-m) * 3jsymbol
            wigner3j_hole_intermediate = wigner_eckart_phase(intermediate_kappa, mj)*wigner3j_numerical2(j_hole, j_intermediate, mjj)
            wigner3j_intermediate_final = wigner_eckart_phase(final_kappa, mj)*wigner3j_numerical2(j_intermediate, j_final, mjj)

            w3j_total = np.double(wigner3j_hole_intermediate)*np.double(wigner3j_intermediate_final)

            # If the 3j symbol is zero we skip the term, if we somehow didn't catch this earlier.
            if(w3j_total != None and np.abs(w3j_total) > 1e-12):
                summed_real[:, unique_state_index] += w3j_total*raw_real[:, col_idx]
                summed_imag[:, unique_state_index] += w3j_total*raw_imag[:, col_idx]
                #print(path_tuple, ("%i/2" % mjj), w3j_total)

            #print(unique_state_index, path_tuple)

    else:
        # This is an invalid mj provided as input and it won't compute any non-zero 3j-symbol.
        # So we throw an error on this input.
        raise ValueError("Invalid mj-value on input to function. No non-zero 3j-symbols will be computed.")

    unique_state_index = 0
    for unique_state in unique_states:
        z = summed_real[:, unique_state_index] + 1j*summed_imag[:, unique_state_index]
        paths_summed[unique_state] = z
        unique_state_index += 1

    return paths_summed


# ==================================================================================================
#
# ==================================================================================================
def XUV_to_abs_or_emi(x, abs_or_emi):
    xlabel_shift = ""
    if(abs_or_emi == "abs"):
        x = x + g_omega_IR
        xlim = xlim + g_omega_IR
        xlabel_shift = " + IR"
    elif(abs_or_emi == "emi"):
        x = x - g_omega_IR
        xlim = xlim - g_omega_IR
        xlabel_shift = " - IR"
    else:
        raise ValueError("abs_or_emi needs to be 'abs' or 'emi' on entry")

    return x, xlabel_shift


# ==================================================================================================
# ## OLD UNUSED
# ==================================================================================================
# class ParticleState:
#     def __init__(self, kappa):
#         self.kappa = kappa
#         self.l = l_from_kappa(kappa)
#         self.j = j_from_kappa(kappa)
#         self.name = "\epsilon " + l_to_str(self.l) + ("_{%i/2}" % (j_from_kappa_int(kappa)))
