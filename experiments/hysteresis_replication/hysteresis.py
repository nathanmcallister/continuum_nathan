#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from camarillo_cc import CamarilloModel
from utils_cc import camarillo_2_webster_params, calculate_transform

# Script parameters
num_points = 2**14
hysteresis_coefficient = 0.05

# Camarillo setup
camarillo_stiffness = np.loadtxt(Path("../../tools/camarillo_stiffness"), delimiter=",")
ka, kb, kt = camarillo_stiffness[0], camarillo_stiffness[1], camarillo_stiffness[2]
cable_positions = [
    ((4, 0), (0, 4), (-4, 0), (0, -4)),
]
segment_stiffness_vals = [(ka, kb)]
cable_stiffness_vals = [(kt, kt, kt, kt)]
segment_lengths = [64]

camarillo_model = CamarilloModel(
    cable_positions,
    segment_stiffness_vals,
    cable_stiffness_vals,
    segment_lengths,
    50,
)

rng = np.random.default_rng(42)

cable_deltas = rng.uniform(-12, 12, (4, num_points))
pos = np.zeros((3, num_points))
no_hist = np.copy(pos)

for i in tqdm(range(num_points)):
    camarillo_params = camarillo_model.forward(cable_deltas[:, i], True)
    webster_params = camarillo_2_webster_params(
        camarillo_params, camarillo_model.segment_lengths
    )
    T = calculate_transform(webster_params)
    pos[:, i] = T[:3, 3]
    no_hist[:, i] = pos[:, i]

    if i != 0:
        pos[:, i] *= 1 - hysteresis_coefficient
        pos[:, i] += hysteresis_coefficient * pos[:, i - 1]

pos += rng.normal(scale=0.5, size=pos.shape)

plt.figure()
plt.scatter(pos[0, :], pos[1, :], alpha=0.3)
plt.title("Camarillo Output Distribution w/ Slack & Hysteresis")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.show()
