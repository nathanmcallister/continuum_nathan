#!/bin/python3
import numpy as np
import time
import utils_data
import continuum_aurora
import continuum_arduino
import kinematics

motor_std = 3
num_motors = 4
num_measurements = 10
sample_period = 0.5

T_aurora_2_model = np.loadtxt("../../tools/T_aurora_2_model", delimiter=",")
T_tip_2_coil = np.loadtxt("../../tools/T_tip_2_coil", delimiter=",")

arduino = continuum_arduino.init_arduino()
aurora = continuum_aurora.init_aurora()

coil_port = "0A"
probe_list = [coil_port]

motor_setpoints = continuum_arduino.load_motor_setpoints("../../tools/motor_setpoints")

container = utils_data.DataContainer()
container.set_date_and_time()
container.prefix = "output/dynamic"

motor_dls = np.random.normal(0, motor_std, (num_motors,num_measurements))
pos = np.nan * np.zeros((3, num_measurements))
tang = np.nan * np.zeros((3, num_measurements))

prev_time = time.perf_counter_ns()
while meas_counter < num_measurements:

    time = time.perf_counter_ns()

    if (time - prev_time) * ns2s >= sample_period:
        transforms = continuum_aurora.get_aurora_transforms(aurora, probe_list, 0.25, 1)
        T = continuum_aurora.get_T_tip_2_model(trans[coil_port], T_aurora_2_model, T_tip_2_coil)

        pos[:, meas_counter] = T[0:3, 3]
        tang[:, meas_counter] = kinematics.dcm_2_tang(T[0:3, 0:3])

        motor_vals = continuum_arduino.one_seg_dl_2_motor_vals(motor_dls[:, meas_counter].tolist(), motor_setpoints)
        continuum_arduino.write_motor_vals(arduino, motor_vals)

        prev_time = time
        meas_counter += 1


container.from_raw_data(container.date, container.time, num_motors, num_measurements, motor_dls, pos, tang)
container.file_export()
