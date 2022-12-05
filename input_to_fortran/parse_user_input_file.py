import numpy as np
from itertools import islice  # Slicing when reading lines from Fortran files.
from distutils.util import strtobool
from input_to_fortran.list_of_user_input_variables import g_user_input_params_list, \
    g_string_parameters, g_float_parameters, g_floatlist_parameters, g_bool_parameters

g_parsed_variables_dict = {}

g_input_file_path = ""
g_line_counter = 0


def raise_error_if_parse_fail(line):
    global g_input_file_path
    global g_line_counter
    print("\n")
    print("ERROR:")
    print("Error parsing ", line, " to key value pair.")
    print("From line %i in file %s" % (g_line_counter, g_input_file_path))
    print("Make sure that user input parameters are properly set and that the variable name exists"
          " in the relcode_py repository.")
    print("\n")
    raise Exception("Couldn't parse key value pair from input file.")

def debug_line_parse(line, key, value):
    print("from line:")
    print(line)
    print("parsed key value pair:", key, value)
    print("type(key):", type(key), "type(value):", type(value))
    print("\n")


def parse_string_to_key_value_pair(line):
    global g_bool_parameters
    global g_float_parameters
    global g_floatlist_parameters
    global g_string_parameters
    # This is the function that decides how
    key = ""
    value = ""
    key_val_split = line.split("=")
    key_str = key_val_split[0]
    val_str = key_val_split[-1]

    if key_str == "" or val_str == "" or key_str not in g_user_input_params_list:
        raise_error_if_parse_fail(line)

    # Key is just the string of the variable name.
    key = key_str

    # Value can be some different datatypes which we handle here.
    # Most will be integers.
    if   key_str in g_float_parameters:
        value = float(val_str)
    elif key_str in g_floatlist_parameters:
        value = [float(x) for x in val_str.split(",")]
    elif key_str == "highest_occupied_orbital":
        val_str = val_str.strip(")")
        val_str = val_str.strip("(")
        val_str = val_str.split(",")
        value = tuple(map(int, val_str))
    elif key_str in g_bool_parameters:
        try:
            value = bool(strtobool(val_str))
        except:
            raise_error_if_parse_fail(line)
    elif key_str in g_string_parameters:
        value = val_str
    else:
        value = int(val_str)

    return key, value


def parse_user_input_file(file_path):
    global g_input_file_path
    global g_line_counter
    global g_parsed_variables_dict
    g_input_file_path = file_path

    print("Parsing user input file %s \n" % (file_path))

    file = open(file_path, "r")
    for line in islice(file, 1, None):
        g_line_counter += 1
        # Remove trailing newline chars
        line[-1].replace("\n","")
        # We skip the line if it starts with a hash (a comment),
        # or if it's empty
        if line[0] == "#" or len(line) == 0 or not line.strip():
            continue

        # remove whitespace
        line = line.replace(" ", "")

        if line.find("#") != -1:
            # Split by comments by just taking the first element in the split string
            line = line.split("#")[0]

        # If we now have a zero length string this line was a comment only line, and we skip.
        if(len(line) == 0):
            continue

        # Else we proceed by saving it as a key value pair.
        key, value = parse_string_to_key_value_pair(line)

        g_parsed_variables_dict[key] = value
        #debug_line_parse(line, key, value)

    file.close()

    if "write_binary_data_for_diag_extension" not in g_parsed_variables_dict.keys():
        g_parsed_variables_dict["write_binary_data_for_diag_extension"] = False
        print("WARNING: Couldn't find write_binary_data_for_diag_extension parameter. Adding it set to False by default.\n"
              "NOTE: Only relevant if running diagonalisation.\n")

    print("Parameters parsed from input file:")
    for param in g_user_input_params_list:
        print(param + " : ", g_parsed_variables_dict[param])


    return g_parsed_variables_dict
