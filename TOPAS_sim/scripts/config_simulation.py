from datetime import datetime

#TODO: Randomising CS_L wrong.

# General Simulation Parameters
GENERAL = {
    'num_simulations': 5000,
    'random_seed': 38,       # Seed for generating the randomised values between ranges
    'template_file': "TOPAS_sim/input/topas_template.inp"
}
GENERAL['output_directory'] = f"TOPAS_sim/simulations/batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_NumSims{GENERAL['num_simulations']}"

# TOPAS Parameters
TOPAS = {
    # Fixed
    'two_theta_start_var': {'value': 5, 'randomize': False},
    'two_theta_end_var': {'value': 150, 'randomize': False},
    'two_theta_step_var': {'value': 0.02, 'randomize': False},
    'background_var': {'value': 1, 'randomize': False},
    'mixture_MAC_var': {'value': 92.040948, 'randomize': False},
    'LP_Factor_var': {'value': 17.0, 'randomize': False},
    'Rp_var': {'value': 173, 'randomize': False},
    'Rs_var': {'value': 173, 'randomize': False},

    'r_exp_var': {'value': 4.06973909, 'randomize': False},
    'r_exp_dash_var': {'value': 4.37166646, 'randomize': False},
    'R_wp_var': {'value': 7.19774449, 'randomize': False},
    'r_wp_dash_var': {'value': 7.73173353, 'randomize': False},
    'r_p_var': {'value': 5.34124753, 'randomize': False},
    'r_p_dash_var': {'value': 5.87721834, 'randomize': False},
    'weighted_Durbin_Watson_var': {'value': 0.782067221, 'randomize': False},
    'gof_var': {'value': 1.76860097, 'randomize': False},

    'Divergence_var': {'value': 1, 'randomize': False},
    'filament_length_var': {'value': 12, 'randomize': False},
    'sample_length_var': {'value': 25, 'randomize': False},
    'receiving_slit_length_var': {'value': 12, 'randomize': False},
    'axial_n_beta_var': {'value': 20, 'randomize': False},
    'axial_del_var': {'value': 0.0053, 'randomize': False},
    'ymin_on_ymax_var': {'value': 0.0001, 'randomize': False},
    'wavelength_distribution_var': {
        'value': [
            {'la': 0.0159, 'lo': 1.534753, 'lh': 3.6854},
            {'la': 0.5791, 'lo': 1.540596, 'lh': 0.437},
            {'la': 0.0762, 'lo': 1.541058, 'lh': 0.6},
            {'la': 0.2417, 'lo': 1.54441, 'lh': 0.52},
            {'la': 0.0871, 'lo': 1.544721, 'lh': 0.62}
        ],
        'randomize': False
    },

    # Randomized
    'Zero_Error_var': {'value': 0, 'randomize': True, 'range': (-0.10, 0.10)},
    'Specimen_Displacement_var': {'value': 0, 'randomize': True, 'range': (-0.15, 0.15)},
    'Absorption_var': {'value': 500, 'randomize': True, 'range': (20, 500)},
    'Slit_Width_var': {'value': 0.368496028, 'randomize': True, 'range': (0.2, 0.45)},
    'primary_soller_angle_var': {'value': 5.75937, 'randomize': True, 'range': (4.0, 7.0)},
    'secondary_soller_angle_var': {'value': 6.62026, 'randomize': True, 'range': (4.0, 7.0)},
}

# Phase Parameters
PHASES = [
    {
        'name': 'Corundum',

        # Fixed
        'r_bragg_var': {'value': 1.77566984, 'randomize': False},
        'Phase_LAC_1_on_cm_var': {'value': 125.994742, 'randomize': False},
        'Phase_Density_g_on_cm3_var': {'value': 3.98833597, 'randomize': False},
        'Al_x_var': {'value': 0, 'randomize': False},
        'Al_y_var': {'value': 0, 'randomize': False},
        'Al_z_var': {'value': 0.35218, 'randomize': False},
        'Al_beq_var': {'value': 0.23356, 'randomize': False},
        'O_x_var': {'value': 0.30603, 'randomize': False},
        'O_y_var': {'value': 0, 'randomize': False},
        'O_z_var': {'value': 0.25, 'randomize': False},
        'O_beq_var': {'value': 0.17408, 'randomize': False},

        # Randomized
        'CS_L_var': {'value': 278.041764, 'randomize': True, 'range': (100, 9999)},
        'CS_G_var': {'value': 1332.83377, 'randomize': True, 'range': (100, 9999)},
        'Strain_L_var': {'value': 0.0140625517, 'randomize': True, 'range': (0.0001, 0.10)},
        'Strain_G_var': {'value': 0.0001, 'randomize': True, 'range': (0.0001, 0.10)},
        'a_var': {'value': 4.758336, 'randomize': True, 'range': (4.7, 4.8)},
        'c_var': {'value': 12.989827, 'randomize': True, 'range': (12.9, 13.1)},
    },
    {
        'name': 'Fluorite',

        # Fixed
        'r_bragg_var': {'value': 2.81476313, 'randomize': False},
        'Phase_LAC_1_on_cm_var': {'value': 301.276541, 'randomize': False},
        'Phase_Density_g_on_cm3_var': {'value': 3.18077903, 'randomize': False},
        'Ca_x_var': {'value': 0, 'randomize': False},
        'Ca_y_var': {'value': 0, 'randomize': False},
        'Ca_z_var': {'value': 0, 'randomize': False},
        'Ca_beq_var': {'value': 0.47587, 'randomize': False},
        'F_x_var': {'value': 0.25, 'randomize': False},
        'F_y_var': {'value': 0.25, 'randomize': False},
        'F_z_var': {'value': 0.25, 'randomize': False},
        'F_beq_var': {'value': 0.67765, 'randomize': False},

        # Randomized
        'CS_L_var': {'value': 513.629301, 'randomize': True, 'range': (100, 9999)},
        'CS_G_var': {'value': 215.48213, 'randomize': True, 'range': (100, 9999)},
        'Strain_L_var': {'value': 0.0142737618, 'randomize': True, 'range': (0.0001, 0.10)},
        'Strain_G_var': {'value': 0.000101123088, 'randomize': True, 'range': (0.0001, 0.10)},
        'a_var': {'value': 5.462971, 'randomize': True, 'range': (5.40, 5.50)},
    },
    {
        'name': 'Zincite',

        # Fixed
        'r_bragg_var': {'value': 1.8027893, 'randomize': False},
        'Phase_LAC_1_on_cm_var': {'value': 276.433402, 'randomize': False},
        'Phase_Density_g_on_cm3_var': {'value': 5.68040088, 'randomize': False},
        'Zn_x_var': {'value': 1/3, 'randomize': False},
        'Zn_y_var': {'value': 2/3, 'randomize': False},
        'Zn_z_var': {'value': 0, 'randomize': False},
        'Zn_beq_var': {'value': 0.49364, 'randomize': False},
        'O_x_var': {'value': 1/3, 'randomize': False},
        'O_y_var': {'value': 2/3, 'randomize': False},
        'O_z_var': {'value': 3/8, 'randomize': False},
        'O_beq_var': {'value': 0.38610, 'randomize': False},

        # Randomized
        'CS_L_var': {'value': 240.726855, 'randomize': True, 'range': (100, 9999)},
        'CS_G_var': {'value': 9991.59547, 'randomize': True, 'range': (100, 9999)},
        'Strain_L_var': {'value': 0.000106573178, 'randomize': True, 'range': (0.0001, 0.10)},
        'Strain_G_var': {'value': 0.000100000009, 'randomize': True, 'range': (0.0001, 0.10)},
        'a_var': {'value': 3.249181, 'randomize': True, 'range': (3.17, 3.27)},
        'c_var': {'value': 5.205846, 'randomize': True, 'range': (5.15, 5.25)},
    },
]