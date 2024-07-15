#!/bin/python3
import numpy as np
import time
import utils_data
import continuum_aurora
import continuum_arduino
import kinematics
from pathlib import Path

ns2s = 10**-9

motor_range = 12
num_motors = 4
num_measurements = 2
sample_period = 0.5

# init filepath
continuum_name = Path(__file__).parent.parent.parent

T_aurora_2_model = np.loadtxt(continuum_name.joinpath("tools","T_aurora_2_model"), delimiter=",")
T_tip_2_coil = np.loadtxt(continuum_name.joinpath("tools","T_tip_2_coil"), delimiter=",")

arduino = continuum_arduino.init_arduino()
aurora = continuum_aurora.init_aurora()

coil_port = "0A"
probe_list = [coil_port]

motor_setpoints = continuum_arduino.load_motor_setpoints(continuum_name.joinpath("tools","motor_setpoints"))

continuum_arduino.write_motor_vals(arduino, motor_setpoints)

container = utils_data.DataContainer()
container.set_date_and_time()
container.prefix = "output/dynamic"

rng = np.random.default_rng()

motor_dls = 2 * motor_range * (rng.random((num_motors, num_measurements)) - 0.5)
pos = np.nan * np.zeros((3, num_measurements))
tang = np.nan * np.zeros((3, num_measurements))

meas_counter = 0
prev_time = time.perf_counter_ns()
while meas_counter < num_measurements:

    current_time = time.perf_counter_ns()

    if (current_time - prev_time) * ns2s >= sample_period:
        try:
            transforms = continuum_aurora.get_aurora_transforms(
                aurora, probe_list, 0.25, 1
            )
            T = continuum_aurora.get_T_tip_2_model(
                transforms[coil_port], T_aurora_2_model, T_tip_2_coil
            )

            pos[:, meas_counter] = T[0:3, 3]
            tang[:, meas_counter] = kinematics.dcm_2_tang(T[0:3, 0:3])
        except:
            pos[:, meas_counter] = np.nan * np.zeros(3)
            tang[:, meas_counter] = np.nan * np.zeros(3)

        motor_vals = continuum_arduino.one_seg_dl_2_motor_vals(
            motor_dls[:, meas_counter].tolist(), motor_setpoints
        )
        continuum_arduino.write_motor_vals(arduino, motor_vals)

        print(f"{meas_counter + 1} of {num_measurements}")
        prev_time = current_time
        meas_counter += 1

time.sleep(0.5)
continuum_arduino.write_motor_vals(arduino, motor_setpoints)
container.from_raw_data(
    container.date, container.time, num_motors, num_measurements, motor_dls, pos, tang
)
container.file_export()
