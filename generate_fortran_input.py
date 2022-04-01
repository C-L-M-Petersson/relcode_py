########################################################################################################################
#
# This code parses the user input file and generates appropriate input files to be read by the Fortran code.
# Typically the Fortran code expects to read the input files from a subfolder "generated_input/" in the same
# folder as the Fortran executable is called from.
# The file with user input is typically in the same folder as the executable is called from,
# and has to be named on the form "<your_name>.relcode_input".
# The variables in the .relcode_input-files must correspond to those
# defined in "list_of_user_input_variables.py".
#
########################################################################################################################
print("\n")
print("===== START Relcode input generation =====")
print("\n")
# First we handle paths to make sure we can import the modules
import sys
import os
import glob

# This is the path where this file is located, it should be relative
# to the input_to_fortran/-subfolder so we can import modules.
relcode_py_path = os.path.dirname(os.path.abspath(__file__))

# This is the folder from which this script is called.
# It is here that the user input file is located.
current_workdir_path = os.path.abspath(os.getcwd())

# Update sys.path with relcode_py_path so we can import local modules
sys.path.append(relcode_py_path)

# Import "local" modules
from input_to_fortran.parse_user_input_file import parse_user_input_file
from input_to_fortran.create_files_for_fortran import create_atom_parameters_file,\
    create_run_parameters_file,  create_file_io_parameters_file, \
    create_knotpoint_sequence_and_box_parameters_file, create_photon_sequence_and_parameters_file, \
    create_generation_complete_file_for_fortran_validation

globbed_filenames = glob.glob(current_workdir_path+"/"+"*.relcode_input")
if len(globbed_filenames) > 1:
    raise Exception("Too many *.relcode_input-files detected! We can only have one input file.")

input_file_name = globbed_filenames[0]  # glob gives us a list so we take the element

# Create directories for Fortran file if they don't exist.
generated_input_path = current_workdir_path+"/generated_input"
if not os.path.exists(generated_input_path):
    try:
        os.mkdir(generated_input_path)
        print("Created directory ", generated_input_path)
    except OSError as error:
        print(error)

#
# Main part of program, parsing user input text file and generating files to be read by Fortran code.
#

parsed_vars_dict = parse_user_input_file(input_file_name)
create_atom_parameters_file(parsed_vars_dict, generated_input_path)
create_run_parameters_file(parsed_vars_dict, generated_input_path)
create_file_io_parameters_file(parsed_vars_dict, current_workdir_path, generated_input_path)
create_knotpoint_sequence_and_box_parameters_file(parsed_vars_dict, generated_input_path)
create_photon_sequence_and_parameters_file(parsed_vars_dict, generated_input_path)
create_generation_complete_file_for_fortran_validation(generated_input_path)

print("===== END Relcode input generation =====")
print("\n")