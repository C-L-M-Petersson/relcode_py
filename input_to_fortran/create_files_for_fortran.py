import os
import numpy as np
from input_to_fortran.list_of_user_input_variables import g_user_input_params_list, \
    g_bool_parameters, g_string_parameters
from input_to_fortran.create_knotpoint_sequence import get_knotpoint_sequence_from_params, write_box_parameters_to_file

g_eV_per_Hartree = 27.211396641308

##################################################################################################
#
##################################################################################################
def write_string_to_file(file, string):
    file.write(string + "\n")
    return


def write_integer_var_comment_and_value(file, var_str, value):
    comment_str = "# " + var_str
    val_str = "%i" % value
    write_string_to_file(file, comment_str)
    write_string_to_file(file, val_str)
    return


def write_double_var_comment_and_value(file, var_str, value):
    comment_str = "# " + var_str
    val_str = "%.5fd0" % value
    write_string_to_file(file, comment_str)
    write_string_to_file(file, val_str)
    return

def create_folder_if_it_doesnt_exist(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
            print("Created directory ", path)
        except OSError as error:
            print(error)

def is_folder_empty(path):
    empty = False
    if os.path.isdir(path):
        if not os.listdir(path):
            empty = True

    return empty

##################################################################################################
#
##################################################################################################
def create_atom_parameters_file(parsed_vars_dict, generated_input_path):

    charge_Z = parsed_vars_dict["nuclear_charge_Z"]
    list_of_orbital_tuples = get_atom_orbitals_list(charge_Z)

    var_list = g_user_input_params_list

    filename = generated_input_path + "/" + "atom_parameters.input"
    file = open(filename, "w")

    orbital_counter = 0

    var = "nuclear_charge_Z"
    write_double_var_comment_and_value(file, var, parsed_vars_dict[var])


    num_orbitals_comment = "# num orbitals"
    write_string_to_file(file, num_orbitals_comment)
    num_orbitals = "%i" % len(list_of_orbital_tuples)
    write_string_to_file(file, num_orbitals)

    var ="number_of_holes"
    write_integer_var_comment_and_value(file, var, parsed_vars_dict[var])

    var = "last_kappa"
    write_integer_var_comment_and_value(file, var, parsed_vars_dict[var])

    # Write all the included orbitals in a sequence here
    write_string_to_file(file, "# Orbitals for atom (n, l, 2j, occupation number)")
    for orbital_tuple in list_of_orbital_tuples:
        write_str = "%i %i %i %i" % (orbital_tuple[0], orbital_tuple[1], orbital_tuple[2], orbital_tuple[3])
        write_string_to_file(file, write_str)

    file.close()

    #print("Wrote to %s" % filename)
    return


##################################################################################################
#
##################################################################################################
def create_run_parameters_file(parsed_vars_dict, generated_input_path):
    filename = generated_input_path + "/" + "run_parameters.input"
    file = open(filename, "w")

    is_any_param_true = False
    for param in g_bool_parameters:
        bool_val = parsed_vars_dict[param]
        value = 0
        if bool_val:
            value = 1
            is_any_param_true = True

        comment_str = param
        write_integer_var_comment_and_value(file, comment_str, value)

    if not is_any_param_true:
        print("\nWARNING! Only running ground state calculation!\n")

    if parsed_vars_dict["run_two_photons"] and not parsed_vars_dict["run_one_photon"]:
        print("\nERROR! run_two_photons set to true while run_one_photon set to false.")
        print("two photons require one photon data.")
        raise Exception("Can't run two photons without running first photon calculation.")

    if parsed_vars_dict["run_diagonalise"] and not parsed_vars_dict["run_forward_only"]:
        print("\n WARNING! Diagonalisation of full RRPAE not fully tested!")

    file.close()
    #print("Wrote to %s" % filename)
    return


##################################################################################################
#
##################################################################################################
def create_file_io_parameters_file(parsed_vars_dict, current_workdir, generated_input_path):
    filename = generated_input_path + "/" + "file_io_parameters.input"

    default_dir_path = current_workdir + "/output"
    # Create folders if necessary
    write_path = parsed_vars_dict["path_to_output_folder"]
    if write_path != "default":
        create_folder_if_it_doesnt_exist(write_path)
    else:
        create_folder_if_it_doesnt_exist(default_dir_path)

    read_path = parsed_vars_dict["path_to_previous_output"]

    if len(read_path) > 1:
        if read_path == "default":
            read_path = default_dir_path

        create_folder_if_it_doesnt_exist(read_path)
        # If the folder is empty and we still try to read from it, things will break.
        # So if the folder is given by user but nothing is in it, we will force this option to zero
        # and give a warning.
        if is_folder_empty(read_path):
            print("WARNING! Empty read folder supplied. Ignoring reading from previous calculation this run.")
            parsed_vars_dict["path_to_previous_output"] = "0"

    # Check that exp en file actually exists if it's supposed to be used.
    exp_en_file = parsed_vars_dict["path_to_experimental_energies"]
    if exp_en_file != "0":
        file_exists = os.path.exists(exp_en_file)
        if not file_exists:
            print("FATAL ERROR: Experimental energies file %s doesn't exist." % exp_en_file)
            raise Exception("Invalid experimental energies file.")



    file = open(filename, "w")
    for param in g_string_parameters:
        val_str = parsed_vars_dict[param]
        if val_str == "default":
            val_str = default_dir_path
            create_folder_if_it_doesnt_exist(default_dir_path)

        comment_str = "# %s" % param
        write_string_to_file(file, comment_str)
        write_string_to_file(file, val_str)

    file.close()



    #print("Wrote to %s" % filename)
    return


##################################################################################################
#
##################################################################################################
def create_knotpoint_sequence_and_box_parameters_file(parsed_vars_dict, generated_input_path):

    Z = parsed_vars_dict["nuclear_charge_Z"]
    first_point = parsed_vars_dict["first_non_zero_point"]
    max_first_nonzero_coord = 0.5/Z

    if first_point > max_first_nonzero_coord:
        print("\n")
        print("WARNING! first_non_zero_point greater than 0.5/Z, for Z = %f" % Z)
        print("first_non_zero_point = ", first_point)
        print("0.5/Z = ", max_first_nonzero_coord)
        print("Consider putting knot points closer to the origin.")
        print("\n")

    print("\nGenerating knotpoint sequence:")
    knotsequence, start_imag_coord = get_knotpoint_sequence_from_params(parsed_vars_dict)
    print("\n")
    knotsequence_filename = generated_input_path + "/" + "knotpoint_sequence.dat"
    np.savetxt(knotsequence_filename, knotsequence, delimiter="    ", fmt='%1.13e')
    #print("Wrote to %s" % knotsequence_filename)

    file_params_filename = generated_input_path + "/" + "box_parameters.input"
    file_params = open(file_params_filename, "w")
    write_box_parameters_to_file(file_params, parsed_vars_dict, start_imag_coord)
    file_params.close()
    #print("Wrote to %s" % file_params_filename)

    return


##################################################################################################
#
##################################################################################################
def create_photon_sequence_and_parameters_file(parsed_vars_dict, generated_input_path):
    en_input_list_filename = parsed_vars_dict["photon_energy_generation"]
    num_xuv         = 0
    num_ir          = 0
    start_omega_eV  = 0
    end_omega_eV    = 0
    simulation_type = ""
    if   en_input_list_filename == "0":
        # We compute the step size as delta_omega = omega_IR/fraction.
        # Note that we translate all input values to atomic units since this is what is appropriate
        # for the Fortran computations.
        simulation_type = "RABITT"
        num_ir = 2

        fraction = parsed_vars_dict["first_photon_step_fraction"]
        omega_IR_eV = parsed_vars_dict["second_photon_energy"]
        omega_IR_au = omega_IR_eV/g_eV_per_Hartree

        delta_omega = omega_IR_au/fraction

        start_omega_eV = parsed_vars_dict["first_photon_energy_start"]
        end_omega_eV = parsed_vars_dict["first_photon_energy_end"]
        in_start_omega_au = start_omega_eV/g_eV_per_Hartree
        in_end_omega_au = end_omega_eV/g_eV_per_Hartree

        # Compute start and end points adhering to step size delta_omega
        tmp_start = 0.0
        count = 0
        while tmp_start < in_start_omega_au:
            tmp_start += delta_omega
            count += 1

        # Go back one step for starting energy to assure we start slightly before selected energy.
        start_energy_au = tmp_start - delta_omega
        num_start_steps = count - 1  # Withdraw one count since we're taking one step back

        tmp_end = 0.0
        count = 0
        while tmp_end < in_end_omega_au:
            tmp_end += delta_omega
            count += 1

        # We will end up in one step above the chosen energy (or exactly hitting it)
        num_end_points = count

        total_photon_points = num_end_points-num_start_steps

        energies_au = [] #np.zeros(total_photon_points,3)
        for i in range(total_photon_points):
            energies_au.append([start_energy_au + i*delta_omega,-omega_IR_au,omega_IR_au])
            #energies_au[i] = [start_energy_au + i*delta_omega]
            #print("%1.13e" % energies_au[i])

        photon_energy_filename = generated_input_path + "/" + "photon_energies.dat"
        np.savetxt(photon_energy_filename, energies_au, fmt='%1.13e')
        #print("Wrote to %s" % photon_energy_filename)
        num_xuv = len(energies_au)
    elif en_input_list_filename == "1":
        simulation_type = "KRAKEN"
        # We compute the step size as delta_omega = omega_IR/fraction.
        # Note that we translate all input values to atomic units since this is what is appropriate
        # for the Fortran computations.

        omega_XUV_plus_IR_eVs = parsed_vars_dict["total_photon_energy"]
        omega_XUV_plus_IR_aus = [o_eV/g_eV_per_Hartree for o_eV in omega_XUV_plus_IR_eVs]

        num_ir = len(omega_XUV_plus_IR_aus)

        fraction = parsed_vars_dict["first_photon_step_fraction"]
        omega_IR_eV = parsed_vars_dict["second_photon_energy"]
        omega_IR_au = omega_IR_eV/g_eV_per_Hartree

        delta_omega = omega_IR_au/fraction

        start_omega_eV = parsed_vars_dict["first_photon_energy_start"]
        end_omega_eV = parsed_vars_dict["first_photon_energy_end"]
        in_start_omega_au = start_omega_eV/g_eV_per_Hartree
        in_end_omega_au = end_omega_eV/g_eV_per_Hartree

        # Compute start and end points adhering to step size delta_omega
        tmp_start = 0.0
        count = 0
        while tmp_start < in_start_omega_au:
            tmp_start += delta_omega
            count += 1

        # Go back one step for starting energy to assure we start slightly before selected energy.
        start_energy_au = tmp_start - delta_omega
        num_start_steps = count - 1  # Withdraw one count since we're taking one step back

        tmp_end = 0.0
        count = 0
        while tmp_end < in_end_omega_au:
            tmp_end += delta_omega
            count += 1

        # We will end up in one step above the chosen energy (or exactly hitting it)
        num_end_points = count

        total_photon_points = num_end_points-num_start_steps

        energies_au = [] #np.zeros(total_photon_points,3)
        for i in range(total_photon_points):
            omega_XUV_au_tmp = start_energy_au + i*delta_omega
            energies_au_tmp  = [omega_XUV_plus_IR_au - omega_XUV_au_tmp for omega_XUV_plus_IR_au in omega_XUV_plus_IR_aus]
            energies_au_tmp.insert(0,omega_XUV_au_tmp)
            energies_au.append(energies_au_tmp)

        photon_energy_filename = generated_input_path + "/" + "photon_energies.dat"
        np.savetxt(photon_energy_filename, energies_au, fmt='%1.13e')
        #print("Wrote to %s" % photon_energy_filename)
        num_xuv = len(energies_au)
    else:
        # This deals with the case where we store the photon energies in a file
        simulation_type = "Custom"

        # Check that the file exists
        file_exists = os.path.exists(en_input_list_filename)
        if not file_exists:
            print("FATAL ERROR: photon energy file %s doesn't exist." % en_input_list_filename)
            raise Exception("Invalid photon energy file.")

        num_ir = -1
        with open(en_input_list_filename) as en_input_list_file:
            i = 0

            au_conversion_factor = 1
            energies_au = []
            for line in en_input_list_file:
                # Remove comments
                if line.find("#") != -1:
                    line = line.split("#")[0]

                # Remove newline
                line = line.replace("\n","")

                # Split into words
                if line.find(" ") != -1:
                    line = line.split(" ")

                # Skip empty lines
                if(len(line) == 0):
                    continue
                print(line)

                if(i == 0):
                    if  (line == "eV"):
                        au_conversion_factor = 1/g_eV_per_Hartree
                    elif(line == "au"):
                        au_conversion_factor = 1
                    else:
                        print("FATAL ERROR: "+line+" is not a valid unit of energy. Please select either eV or au.")
                        raise Exception("Invalid unit of photon energy file.")
                else:
                    energies_au.append([au_conversion_factor*float(x) for x in line])

                if  (i   ==  0):
                    pass
                elif(num_ir == -1):
                    num_ir = len(line)-1
                elif(num_ir != len(line)-1):
                    print("FATAL ERROR: photon energy file %s does not contain the same number of energies per line." % en_input_list_filename)
                    raise Exception("Invalid photon energy file.")

                i+=1

        photon_energy_filename = generated_input_path + "/" + "photon_energies.dat"
        np.savetxt(photon_energy_filename, energies_au, fmt='%1.13e')

        start_omega_eV = min([es[0] for es in energies_au]) # We don't assume that these are ordered
        end_omega_eV   = max([es[0] for es in energies_au]) # (although I see no reason for them not to be)
        num_xuv        = len(energies_au)
        # Might not make sense depending on the input data:
        omega_IR_eV = parsed_vars_dict["second_photon_energy"]
        omega_IR_au = omega_IR_eV/g_eV_per_Hartree
        fraction    = round(omega_IR_au*num_xuv/(end_omega_eV-start_omega_eV))


    photon_params_filename = generated_input_path + "/" + "photon_parameters.input"
    file = open(photon_params_filename, "w")

    write_integer_var_comment_and_value(file, "number of first photon energies" , num_xuv)
    write_integer_var_comment_and_value(file, "number of second photon energies", num_ir )

    file.close()
    #print("Wrote to %s" % photon_params_filename)

    print("\n")
    print(simulation_type+" photon input data:")
    print("For first photon interval between %.3f eV and %.3f eV, there are %i "
          "ir energies for each of the %i xuv points."
          % (start_omega_eV, end_omega_eV, num_ir, num_xuv))
    if(simulation_type=="RABITT"): print("omega_IR = %.3f eV" % omega_IR_eV)
    if(simulation_type=="KRAKEN"):
        if num_ir == 1:
            e_final = parsed_vars_dict["total_photon_energy"]
            print("omega_XUV+omega_IR = %.3f eV" % e_final[0])
        else:
            print("omega_XUV+omega_IR in {"+','.join([str(val)+"eV" for val in parsed_vars_dict["total_photon_energy"]])+"}")
    print("\n")

def create_generation_complete_file_for_fortran_validation(generated_input_path):
    filename = generated_input_path + "/" + "generation_complete.input"

    file = open(filename, "w")
    write_string_to_file(file, "# generation complete - fortran checks if this file exists")

    return


def remove_previous_generation_complete_file(generated_input_path):
    filename = generated_input_path + "/" + "generation_complete.input"
    file_exists = os.path.exists(filename)
    if(file_exists):
        os.remove(filename)

    return


def get_atom_orbitals_list(charge_Z):
    # The Fortran program uses a list of orbitals in a (n, l, 2j, occupation_number) format.
    # This function provides that kind of list to the generator script.
    # You can extend this whatever way necessary, for example when calculation some ion.
    # The Fortran program calculates the long range charge as
    # lr_charge = charge_Z-sum(occupation_numbers)+1.0
    # So specifying Z and orbitals with proper occupation numbers should suffice for running ions.

    # He = 1s^2
    helium_orbitals = [
        (1, 0, 1, 2),
    ]

    # Ne = [He] 2s^2 2p^6
    neon_orbitals = helium_orbitals + [
        (2, 0, 1, 2),
        (2, 1, 1, 2),
        (2, 1, 3, 4),
    ]

    # Ar = [Ne] 3s^2 3p^6
    argon_orbitals = neon_orbitals + [
        (3, 0, 1, 2),
        (3, 1, 1, 2),
        (3, 1, 3, 4)
    ]

    # Kr = [Ar] 3d^10 4s^2 4p^6
    krypton_orbitals = argon_orbitals + [
        (3, 2, 3, 4),
        (3, 2, 5, 6),
        (4, 0, 1, 2),
        (4, 1, 1, 2),
        (4, 1, 3, 4)
    ]

    # Xe = [Kr] 4d^10 5s^2 5p^6
    xenon_orbitals = krypton_orbitals + [
        (4, 2, 3, 4),
        (4, 2, 5, 6),
        (5, 0, 1, 2),
        (5, 1, 1, 2),
        (5, 1, 3, 4)
    ]

    # Rn = [Xe] 4f^14 5d^10 6s^2 6p^6
    # Since we now ad 4f before n=5 orbitals we
    # just add first two rows from xenon orbitals first.
    radon_orbitals = krypton_orbitals + [
        (4, 2, 3, 4),
        (4, 2, 5, 6),
        (4, 3, 5, 6),
        (4, 3, 7, 8),
        (5, 0, 1, 2),
        (5, 1, 1, 2),
        (5, 1, 3, 4),
        (5, 2, 3, 4),
        (5, 2, 5, 6),
        (6, 0, 1, 2),
        (6, 1, 1, 2),
        (6, 1, 3, 4)
    ]

    if charge_Z == 2.0:
        return helium_orbitals
    elif charge_Z == 10.0:
        return neon_orbitals
    elif charge_Z == 18.0:
        return argon_orbitals
    elif charge_Z == 36.0:
        return krypton_orbitals
    elif charge_Z == 54.0:
        return xenon_orbitals
    elif charge_Z == 86.0:
        return radon_orbitals
    else:
        print("\nFATAL ERROR: Nuclear charge Z = %.1f not supported!" % charge_Z)
        print("You can add your own set of orbitals to get_atom_orbitals_list() in file %s \n"
              % __file__)
        raise Exception("FATAL ERROR: Nuclear charge Z = %.1f not supported!" % charge_Z,
        "You can add your own set of orbitals to get_atom_orbitals_list() in file %s"
        % __file__)


