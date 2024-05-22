#!/bin/python3
import numpy as np
import time
import utils_data
import kinematics
import continuum_aurora
import continuum_arduino

num_points = 17
num_measurements = 2 * num_points * (num_points - 1)
wait_time = 2
cable_max = 12

T_aurora_2_model = np.loadtxt("../../tools/T_aurora_2_model", delimiter=",")
T_tip_2_coil = np.loadtxt("../../tools/T_tip_2_coil", delimiter=",")

motor_setpoints = continuum_arduino.load_motor_setpoints("../../tools/motor_setpoints")

arduino = continuum_arduino.init_arduino()
aurora = continuum_aurora.init_aurora()

rng = np.random.default_rng()
rand_values = 2 * cable_max * (rng.random((2, num_points)) - 0.5)

cable_displacements = np.concatenate([-rand_values, rand_values], axis=0)

continuum_arduino.write_motor_vals(arduino, motor_setpoints)
time.sleep(wait_time)


num_range = np.array(list(range(num_points)))
cable_deltas = np.zeros((4, num_measurements))
pos = np.nan * np.zeros((3, num_measurements))
tang = np.nan * np.zeros((3, num_measurements))

print("i", "j", "#", "idx")
for i in range(num_points):
    desired_displacement = cable_displacements[:, i]
    other_displacements = cable_displacements[
        :, np.random.permutation(num_range[num_range != i])
    ]

    for j in range(num_points - 1):
        idx = 2 * (num_points - 1) * i + 2 * j
        print(i, j, 0, idx)
        motor_vals = continuum_arduino.one_seg_dl_2_motor_vals(
            list(other_displacements[:, j]), motor_setpoints
        )
        continuum_arduino.write_motor_vals(arduino, motor_vals)
        time.sleep(wait_time)

        transforms = continuum_aurora.get_aurora_transforms(aurora, ["0A"])
        T = continuum_aurora.get_T_tip_2_model(
            transforms["0A"], T_aurora_2_model, T_tip_2_coil
        )
        cable_deltas[:, idx] = other_displacements[:, j]
        pos[:, idx] = T[0:3, 3]
        tang[:, idx] = kinematics.dcm_2_tang(T[0:3, 0:3])

        idx = 2 * (num_points - 1) * i + 2 * j + 1
        print(i, j, 1, idx)
        motor_vals = continuum_arduino.one_seg_dl_2_motor_vals(
            list(desired_displacement), motor_setpoints
        )
        continuum_arduino.write_motor_vals(arduino, motor_vals)
        time.sleep(wait_time)

        transforms = continuum_aurora.get_aurora_transforms(aurora, ["0A"])
        T = continuum_aurora.get_T_tip_2_model(
            transforms["0A"], T_aurora_2_model, T_tip_2_coil
        )
        cable_deltas[:, idx] = desired_displacement
        pos[:, idx] = T[0:3, 3]
        tang[:, idx] = kinematics.dcm_2_tang(T[0:3, 0:3])


continuum_arduino.write_motor_vals(arduino, motor_setpoints)
container = utils_data.DataContainer()
container.prefix = "output/data"
container.set_date_and_time()
container.from_raw_data(
    container.date, container.time, 4, num_measurements, cable_deltas, pos, tang
)
container.file_export()
