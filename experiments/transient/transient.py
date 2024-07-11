#!/bin/python3
import numpy as np
import time
from continuum_arduino import ContinuumArduino
from continuum_aurora import ContinuumAurora
from kinematics import dcm_2_tang
from utils_data import DataContainer

frequency = 40
seconds = 4
points = 25

num_meas = frequency * seconds * points
pos = np.zeros((3, num_meas))
tang = np.zeros((3, num_meas))

T_aurora_2_model = np.loadtxt("../../tools/T_aurora_2_model", delimiter=",")
T_tip_2_coil = np.loadtxt("../../tools/T_tip_2_coil", delimiter=",")

arduino = ContinuumArduino()
aurora = ContinuumAurora(T_aurora_2_model, T_tip_2_coil)

rng = np.random.default_rng(123)
cable_dls = np.concatenate(
    [np.zeros((4, 1)), 24 * rng.random((4, points - 2)) - 12, np.zeros((4, 1))], axis=1
)
repeated_cable_dls = np.repeat(cable_dls, frequency * seconds, axis=1)

arduino.write_dls(np.zeros(4))
time.sleep(2)
aurora.serial_port.flush()


for i in range(num_meas):
    raw_transforms = aurora.read_aurora_transforms(["0A"])
    print(
        f"Point {i+1:04}/{num_meas}, Command {int(i / (frequency * seconds)) + 1:02}/{points}",
        end="\r",
    )
    try:
        T_tip_2_model = aurora.get_T_tip_2_model(raw_transforms["0A"])
        pos[:, i] = T_tip_2_model[:3, 3]
        tang[:, i] = dcm_2_tang(T_tip_2_model[:3, :3])
    except:
        pos[:, i] = np.nan
        tang[:, i] = np.nan
    arduino.write_dls(repeated_cable_dls[:, i])

print()

container = DataContainer()
container.set_date_and_time()
container.from_raw_data(
    container.date, container.time, 4, num_meas, repeated_cable_dls, pos, tang
)
container.prefix = "output/data"
container.file_export()
