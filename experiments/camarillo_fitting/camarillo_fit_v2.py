#!/bin/python3
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from camarillo_cc import CamarilloModel
from utils_cc import camarillo_2_webster_params, calculate_transform
from utils_data import DataContainer
from kinematics import dcm_2_tang

training_data = "./kinematic_2024_07_16_21_33_12.dat"
data = DataContainer()
data.file_import(training_data)
cable_deltas, pos, tang = data.to_numpy()

cable_positions = [((4, 0), (0, 4), (-4, 0), (0, -4))]
camarillo_stiffness = np.loadtxt(Path("../../tools/camarillo_stiffness"), delimiter=",")
segment_stiffness_vals = [(camarillo_stiffness[0], camarillo_stiffness[1])]
cable_stiffness_vals = [(10000,) * 4]
segment_lengths = [64]

camarillo_model = CamarilloModel(
    cable_positions, segment_stiffness_vals, cable_stiffness_vals, segment_lengths, 50
)

x0 = np.concatenate([(camarillo_stiffness[:2] / 1000).tolist(), np.zeros(4)])
counter = 0


def optim_function(
    x: np.ndarray,
    camarillo_model: CamarilloModel,
    cable_deltas: np.ndarray,
    pos: np.ndarray,
):
    global counter
    current = time.time()
    num_measurements = pos.shape[1]
    dls = cable_deltas + x[2:].reshape((4, 1))
    segment_stiffness_vals = [(1000 * x[0], 1000 * x[1])]
    camarillo_model.segment_stiffness_vals = segment_stiffness_vals

    camarillo_model.update_matrices()
    model_pos = np.zeros((3, num_measurements))
    counter += 1
    for i in tqdm(range(num_measurements), desc=f"Round {counter}"):
        camarillo_params = camarillo_model.forward(dls[:, i], True)
        webster_params = camarillo_2_webster_params(
            camarillo_params, camarillo_model.segment_lengths
        )

        T = calculate_transform(webster_params)
        model_pos[:, i] = T[:3, 3]
    error = np.sqrt(((model_pos - pos) ** 2).mean())
    print(
        f"iteration: {time.time() - current:.3f}s, {error:.3f}mm, {', '.join([f'{y:.3f}' for y in x.tolist()])}"
    )
    return error


results = minimize(
    lambda x: optim_function(x, camarillo_model, cable_deltas[:, :], pos[:, :]),
    x0,
    method="nelder-mead",
)

print(results)

np.savetxt("x.dat", results.x, delimiter=",")
