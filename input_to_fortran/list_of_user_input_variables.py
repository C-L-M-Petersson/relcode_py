# This is a hardcoded list of strings corresponding to the parameters in the input file.
# If this is not a 1-to-1 correspondence things will probably not work.
# If additional input parameters are added to the input file they should also be added here.
# Then they also need to be handled in the proper way after parsing, of course.
g_user_input_params_list = [
    "nuclear_charge_Z",
    #"highest_occupied_orbital", # Use explicit map instead
    "number_of_holes",
    "last_kappa",
    "run_one_photon",
    "run_forward_only",
    "run_two_photons",
    "run_diagonalise",
    "write_diag_eigenvectors",
    "write_diag_coefficients",
    "write_binary_data_for_diag_extension",
    "path_to_output_folder",
    "path_to_previous_output",
    "path_to_experimental_energies",
    "bspline_order_k",
    "total_number_of_bsplines",
    "grid_start_point",
    "grid_end_point",
    "first_non_zero_point",
    "second_point_multiplier",
    "end_point_of_inner_region",
    "inner_region_number_of_exponential_points",
    "mid_region_number_of_linear_points",
    "ecs_region_starting_point",
    "ecs_region_number_of_points",
    "ecs_end_imaginary_coordinate",
    "photon_energy_generation",
    "first_photon_energy_start",
    "first_photon_energy_end",
    "first_photon_step_fraction",
    "second_photon_energy",
    "total_photon_energy"
]

# We define what types the parameters have for easy parsing conditions
g_bool_parameters      = ["run_one_photon",
                          "run_forward_only",
                          "run_two_photons",
                          "run_diagonalise",
                          "write_diag_eigenvectors",
                          "write_diag_coefficients",
                          "write_binary_data_for_diag_extension"
                         ]

g_float_parameters     = ["nuclear_charge_Z",
                          "grid_start_point",
                          "grid_end_point",
                          "first_non_zero_point",
                          "second_point_multiplier",
                          "end_point_of_inner_region",
                          "ecs_region_starting_point",
                          "ecs_end_imaginary_coordinate",
                          "first_photon_energy_start",
                          "first_photon_energy_end",
                          "second_photon_energy"
                         ]

g_floatlist_parameters = ["total_photon_energy"
                         ]

g_string_parameters    = ["path_to_output_folder",
                          "path_to_previous_output",
                          "path_to_experimental_energies",
                          "photon_energy_generation"
                         ]
