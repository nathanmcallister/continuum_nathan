#!/bin/python3
import numpy as np
import scipy.optimize as opt
import utils_cc
import utils_data
import camarillo_cc
import kinematics

cable_youngs_modulus = 192e9
cable_area = np.pi * (0.018 * 0.0254 / 2)**2
cable_stiffness = cable_youngs_modulus * cable_area

def optimization_function(x: np.ndarray, cable_deltas: np.ndarray, pos: np.ndarray, tang: np.ndarray, tang_weight: float = 10) -> float:
    """ x is comprised of the following parameters:
        axial stiffness,
        bending stiffness,
        cable_theta,
        """

    cable_pos = [((4*np.cos(x[2]), 4*np.sin(x[2])), )]
    segment_stiffness_vals = [(x[0], x[1])]
    cable_stiffness_vals = [(cable_stiffness,)]
    segment_lengths = [64]
    additional_cable_length = 30


    model_pos = np.zeros(pos.shape)
    model_tang = np.zeros(tang.shape)

    D, K_m_inv, L_0, L_t, K_t_inv = camarillo_cc.get_camarillo_matrices(cable_pos, segment_stiffness_vals, cable_stiffness_vals, segment_lengths, additional_cable_length, [1])


    C_m = np.matmul(
        np.matmul(np.matmul(np.transpose(D), L_0), K_m_inv), D
    ) + np.matmul(L_t, K_t_inv)

    C_m_inv = np.linalg.inv(C_m)

    # q = Ay
    A = np.matmul(np.matmul(K_m_inv, D), C_m_inv)
    q = np.matmul(A, cable_deltas.reshape((1, -1)))

    for i in range(q.shape[1]):
        webster_params = camarillo_cc.camarillo_2_webster_params(q[:, i], segment_lengths)
        T = utils_cc.calculate_transform(webster_params[0])

        model_pos[:, i] = T[0:3, 3]
        model_tang[:, i] = kinematics.dcm_2_tang(T[0:3, 0:3])


    pos_error = (np.linalg.norm(model_pos - pos, axis=0)**2).mean()
    #tang_error = tang_weight * (np.linalg.norm(model_tang - tang, axis=0)**2).mean()

    #return np.sqrt(pos_error + tang_error)
    return np.sqrt(pos_error)

datafile = "output/data_2024_04_21_14_23_39.dat"
axial_stiffness_guess = 500
bending_stiffness_guess = 5000
segment_length_guess = 64
cable_positions = [(4, 0), (0, 4), (-4, 0), (0, -4)]

container = utils_data.DataContainer()
container.file_import(datafile)

num_cables = container.num_cables
num_measurements = container.num_measurements
measurements_per_cable = int(num_measurements / num_cables)

cable_deltas = np.nan * np.zeros(num_measurements)
pos = np.nan * np.zeros((3, num_measurements))
tang = np.nan * np.zeros((3, num_measurements))

for i in range(num_measurements):
    pos[:, i] = container.outputs[i][0:3]
    tang[:, i] = container.outputs[i][3:]
    cable_deltas[i] = container.inputs[i][int(i / measurements_per_cable)]

results = []

for i in range(4):
    theta_0 = np.arctan2(cable_positions[i][1], cable_positions[i][0])
    x0 = np.array([axial_stiffness_guess, bending_stiffness_guess, theta_0])

    lower_range = measurements_per_cable * i
    upper_range = measurements_per_cable * (i + 1)

    op = lambda x : optimization_function(x, cable_deltas[lower_range:upper_range], pos[:, lower_range:upper_range], tang[:, lower_range:upper_range])

    results.append(opt.minimize(op, x0, method='BFGS'))


success_counter = 0
ka = 0
kb = 0

for res in results:
    if res['status'] == 0:
        success_counter += 1
        ka += res['x'][0]
        kb += res['x'][1]

ka /= success_counter
kb /= success_counter

with open('../../tools/camarillo_stiffness', 'w') as f:
    f.write(f"{ka},{kb},{cable_stiffness}")
