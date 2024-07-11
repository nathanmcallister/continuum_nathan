#!/bin/python3
import numpy as np
import time
import utils_data
from continuum_aurora import ContinuumAurora
from continuum_arduino import ContinuumArduino
import kinematics

ns2s = 10**-9

motor_std = 4
motor_range = 13
num_motors = 4
num_measurements = 2**14
sample_period = 4

T_aurora_2_model = np.loadtxt("../../tools/T_aurora_2_model", delimiter=",")
T_tip_2_coil = np.loadtxt("../../tools/T_tip_2_coil", delimiter=",")

arduino = ContinuumArduino()
aurora = ContinuumAurora(T_aurora_2_model, T_tip_2_coil)

probe = "0A"
probe_list = [probe]

arduino.write_dls(np.zeros(4))
time.sleep(2)

rng = np.random.default_rng()

motor_dls = np.concatenate(
    (
        np.zeros((num_motors, 1)),
        2 * motor_range * (rng.random((num_motors, num_measurements)) - 0.5),
    ),
    axis=1,
)
pos = np.nan * np.zeros((3, num_measurements))
tang = np.nan * np.zeros((3, num_measurements))

container = utils_data.DataContainer()
container.set_date_and_time()
container.prefix = "output/kinematic"

meas_counter = 0
prev_time = time.perf_counter_ns()
while meas_counter < num_measurements:

    current_time = time.perf_counter_ns()

    if (current_time - prev_time) * ns2s >= sample_period:
        try:
            transforms = aurora.get_aurora_transforms(probe_list)
            T = aurora.get_T_tip_2_model(transforms[probe])

            pos[:, meas_counter] = T[0:3, 3]
            tang[:, meas_counter] = kinematics.dcm_2_tang(T[0:3, 0:3])
        except:
            pos[:, meas_counter] = np.nan * np.zeros(3)
            tang[:, meas_counter] = np.nan * np.zeros(3)

        arduino.write_dls(motor_dls[:, meas_counter + 1])

        prev_time = current_time
        print(
            f"Measurement {meas_counter+1:0{int(np.ceil(np.log10(num_measurements + 1)))}}/{num_measurements}",
            end="\r",
        )
        meas_counter += 1
print()
motor_dls = motor_dls[:, :-1]
time.sleep(2)
arduino.write_dls(np.zeros(4))
container.from_raw_data(
    container.date, container.time, num_motors, num_measurements, motor_dls, pos, tang
)
container.file_export()
