#!/bin/python3
import numpy as np
import time
from continuum_arduino import ContinuumArduino
from continuum_aurora import ContinuumAurora
import kinematics
import utils_data

max_displacement = 12
angular_steps = 128
wait = 4

T_aurora_2_model = np.loadtxt("../../tools/T_aurora_2_model", delimiter=",")
T_tip_2_coil = np.loadtxt("../../tools/T_tip_2_coil", delimiter=",")

phi = (np.arange(0, angular_steps) * 2 * np.pi / angular_steps).reshape((1, -1))

dls = -max_displacement * np.concatenate(
    [np.cos(phi), np.sin(phi), -np.cos(phi), -np.sin(phi)], axis=0
)

arduino = ContinuumArduino()
aurora = ContinuumAurora(T_aurora_2_model, T_tip_2_coil)

positions = np.nan * np.zeros((3, angular_steps))

arduino.write_dls(np.zeros(4))
time.sleep(wait)

for i in range(angular_steps):
    arduino.write_dls(dls[:, i])
    time.sleep(wait)
    transforms = aurora.get_aurora_transforms(["0A"])
    T_tip_2_model = aurora.get_T_tip_2_model(transforms["0A"])
    positions[:, i] = T_tip_2_model[:3, 3]

np.savetxt("output/max_sweep.dat", delimiter=",")
