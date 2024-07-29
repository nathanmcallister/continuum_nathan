#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import time
from utils_cc import camarillo_2_webster_params, calculate_transform
from camarillo_cc import CamarilloModel

camarillo_stiffness = np.loadtxt("../../tools/camarillo_stiffness", delimiter=",")
ka, kb, kt = camarillo_stiffness[0], camarillo_stiffness[1], camarillo_stiffness[2]
cable_positions = [
    (
        (4, 0),
        (0, 4),
        (-4, 0),
        (0, -4),
    )
]
segment_stiffness_vals = [(ka, kb)]
cable_stiffness_vals = [(kt, kt, kt, kt)]
segment_lengths = [64]

spine = CamarilloModel(
    cable_positions,
    segment_stiffness_vals,
    cable_stiffness_vals,
    segment_lengths,
    50,
)

max_displacement = 12
angular_steps = 128
phi = (np.arange(0, angular_steps) * 2 * np.pi / angular_steps).reshape((1, -1))

dls = -max_displacement * np.concatenate(
    [np.cos(phi), np.sin(phi), -np.cos(phi), -np.sin(phi)], axis=0
)

phi = phi.flatten()

positions = np.zeros((3, angular_steps))
angle = np.zeros(angular_steps)

for i in range(angular_steps):

    camarillo_params = spine.forward(dls[:, i], True)
    webster_params = camarillo_2_webster_params(camarillo_params, spine.segment_lengths)

    T = calculate_transform(webster_params)

    positions[:, i] = T[:3, 3]
    angle[i] = np.arctan2(positions[1, i], positions[0, i])

plt.figure()
plt.plot(phi, np.unwrap(angle))
plt.show()
