#!/bin/python3
import numpy as np
import time
from continuum_arduino import ContinuumArduino
from continuum_aurora import ContinuumAurora
import kinematics
import utils_data
from pathlib import Path

max_displacement = 12
angular_steps = 128
repetitions = 8
wait = 0.5

# init filepath
continuum_name = Path(__file__).parent.parent.parent

T_aurora_2_model = np.loadtxt(
    continuum_name.joinpath("tools", "T_aurora_2_model"), delimiter=","
)
T_tip_2_coil = np.loadtxt(
    continuum_name.joinpath("tools", "T_tip_2_coil"), delimiter=","
)

phi = -(np.arange(0, angular_steps) * 2 * np.pi / angular_steps).reshape((1, -1))

dls = -max_displacement * np.concatenate(
    [np.cos(phi), np.sin(phi), -np.cos(phi), -np.sin(phi)], axis=0
)

arduino = ContinuumArduino()
aurora = ContinuumAurora(T_aurora_2_model, T_tip_2_coil)

positions = np.nan * np.zeros((3, repetitions * angular_steps))

arduino.write_dls(np.zeros(4))
time.sleep(wait)
for i in range(repetitions):
    for j in range(angular_steps):
        print(
            f"Repetition {i+1}/{repetitions}, Step {j+1:03}/{angular_steps}", end="\r"
        )
        arduino.write_dls(dls[:, j])
        time.sleep(wait)
        transforms = aurora.get_aurora_transforms(["0A"])
        T_tip_2_model = aurora.get_T_tip_2_model(transforms["0A"])
        idx = angular_steps * i + j
        positions[:, idx] = T_tip_2_model[:3, 3]

arduino.write_dls(np.zeros(4))
print()
np.savetxt(
    Path(__file__).parent.joinpath("output", "multi_sweep_backwards.dat"),
    positions,
    delimiter=",",
)
