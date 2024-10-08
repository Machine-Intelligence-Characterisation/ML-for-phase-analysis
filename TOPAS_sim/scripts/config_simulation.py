from datetime import datetime

# General Simulation Parameters
GENERAL = {
    'num_simulations': 10000,
    'random_seed': 73
}
GENERAL['output_directory'] = f"TOPAS_sim/simulations/batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_NumSims{GENERAL['num_simulations']}"

# TOPAS Parameters
TOPAS = {
    # Fixed
    'template_file': "TOPAS_sim/input/topas_template.inp",
    'two_theta_start': {'value': 5, 'randomize': False},
    'two_theta_end': {'value': 150, 'randomize': False},
    'two_theta_step': {'value': 0.02, 'randomize': False},
    'background': {'value': 1, 'randomize': False},
    'mixture_MAC': {'value': 92.040948, 'randomize': False},
    'LP_Factor': {'value': 17.0, 'randomize': False},
    'Rp': {'value': 173, 'randomize': False},
    'Rs': {'value': 173, 'randomize': False},
    'Divergence': {'value': 1, 'randomize': False},
    'filament_length': {'value': 12, 'randomize': False},
    'sample_length': {'value': 25, 'randomize': False},
    'receiving_slit_length': {'value': 12, 'randomize': False},
    'axial_n_beta': {'value': 20, 'randomize': False},
    'axial_del': {'value': 0.0053, 'randomize': False},
    'ymin_on_ymax': {'value': 0.0001, 'randomize': False},
    'wavelength_distribution': {
        'value': [
            (0.0159, 1.534753, 3.6854),
            (0.5791, 1.540596, 0.437),
            (0.0762, 1.541058, 0.6),
            (0.2417, 1.54441, 0.52),
            (0.0871, 1.544721, 0.62)
        ],
        'randomize': False
    },

    # Randomized
    'Zero_Error': {'value': 0, 'randomize': True, 'range': (-0.10, 0.10)},
    'Specimen_Displacement': {'value': 0, 'randomize': True, 'range': (-0.15, 0.15)},
    'Absorption': {'value': 500, 'randomize': True, 'range': (20, 500)},
    'Slit_Width': {'value': 0.368496028, 'randomize': True, 'range': (0.2, 0.45)},
    'primary_soller_angle': {'value': 5.75937, 'randomize': True, 'range': (4.0, 7.0)},
    'secondary_soller_angle': {'value': 6.62026, 'randomize': True, 'range': (4.0, 7.0)},
}

# Phase Parameters
PHASES = [
    {
        'name': 'Corundum',

        # Fixed
        'r_bragg': {'value': 1.77566984, 'randomize': False},
        'Phase_LAC_1_on_cm': {'value': 125.994742, 'randomize': False},
        'Phase_Density_g_on_cm3': {'value': 3.98833597, 'randomize': False},
        'Al_x': {'value': 0, 'randomize': False},
        'Al_y': {'value': 0, 'randomize': False},
        'Al_z': {'value': 0.35218, 'randomize': False},
        'Al_beq': {'value': 0.23356, 'randomize': False},
        'O_x': {'value': 0.30603, 'randomize': False},
        'O_y': {'value': 0, 'randomize': False},
        'O_z': {'value': 0.25, 'randomize': False},
        'O_beq': {'value': 0.17408, 'randomize': False},

        # Randomized
        'CS_L': {'value': 278.041764, 'randomize': True, 'range': (100, 9999)},
        'CS_G': {'value': 1332.83377, 'randomize': True, 'range': (100, 9999)},
        'Strain_L': {'value': 0.0140625517, 'randomize': True, 'range': (0.0001, 0.10)},
        'Strain_G': {'value': 0.0001, 'randomize': True, 'range': (0.0001, 0.10)},
        'a': {'value': 4.758336, 'randomize': True, 'range': (4.7, 4.8)},
        'c': {'value': 12.989827, 'randomize': True, 'range': (12.9, 13.1)},
        'scale': {'value': 0, 'randomize': True, 'range': (0, 1)},
    },
    {
        'name': 'Fluorite',

        # Fixed
        'r_bragg': {'value': 2.81476313, 'randomize': False},
        'Phase_LAC_1_on_cm': {'value': 301.276541, 'randomize': False},
        'Phase_Density_g_on_cm3': {'value': 3.18077903, 'randomize': False},
        'Ca_x': {'value': 0, 'randomize': False},
        'Ca_y': {'value': 0, 'randomize': False},
        'Ca_z': {'value': 0, 'randomize': False},
        'Ca_beq': {'value': 0.47587, 'randomize': False},
        'F_x': {'value': 0.25, 'randomize': False},
        'F_y': {'value': 0.25, 'randomize': False},
        'F_z': {'value': 0.25, 'randomize': False},
        'F_beq': {'value': 0.67765, 'randomize': False},

        # Randomized
        'CS_L': {'value': 513.629301, 'randomize': True, 'range': (100, 9999)},
        'CS_G': {'value': 215.48213, 'randomize': True, 'range': (100, 9999)},
        'Strain_L': {'value': 0.0142737618, 'randomize': True, 'range': (0.0001, 0.10)},
        'Strain_G': {'value': 0.000101123088, 'randomize': True, 'range': (0.0001, 0.10)},
        'a': {'value': 5.462971, 'randomize': True, 'range': (5.40, 5.50)},
        'scale': {'value': 0, 'randomize': True, 'range': (0, 1)},
    },
    {
        'name': 'Zincite',

        # Fixed
        'r_bragg': {'value': 1.8027893, 'randomize': False},
        'Phase_LAC_1_on_cm': {'value': 276.433402, 'randomize': False},
        'Phase_Density_g_on_cm3': {'value': 5.68040088, 'randomize': False},
        'Zn_x': {'value': 1/3, 'randomize': False},
        'Zn_y': {'value': 2/3, 'randomize': False},
        'Zn_z': {'value': 0, 'randomize': False},
        'Zn_beq': {'value': 0.49364, 'randomize': False},
        'O_x': {'value': 1/3, 'randomize': False},
        'O_y': {'value': 2/3, 'randomize': False},
        'O_z': {'value': 3/8, 'randomize': False},
        'O_beq': {'value': 0.38610, 'randomize': False},

        # Randomized
        'CS_L': {'value': 240.726855, 'randomize': True, 'range': (100, 9999)},
        'CS_G': {'value': 9991.59547, 'randomize': True, 'range': (100, 9999)},
        'Strain_L': {'value': 0.000106573178, 'randomize': True, 'range': (0.0001, 0.10)},
        'Strain_G': {'value': 0.000100000009, 'randomize': True, 'range': (0.0001, 0.10)},
        'a': {'value': 3.249181, 'randomize': True, 'range': (3.17, 3.27)},
        'c': {'value': 5.205846, 'randomize': True, 'range': (5.15, 5.25)},
        'scale': {'value': 0, 'randomize': True, 'range': (0, 1)},
    },
]

# Machine Learning Parameters
ML = {
    'train_test_split': 0.8,
    'random_state': 42,
}