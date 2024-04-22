#!/bin/python3
import numpy as np
import time
import continuum_aurora
import continuum_arduino
import utils_data
import mike_cc
import camarillo_cc
import utils_cc
import kinematics

cable_positions = [(8, 0), (0, 8), (-8, 0), (0, -8)]
theta_steps = 24
samples_per_position = 5
num_positions = theta_steps * 4
num_measurements = num_positions * samples_per_position
transform_attempts = 5

arduino = continuum_arduino.init_arduino()
aurora = continuum_aurora.init_aurora()
probe_list = ["0A"]

T_aurora_2_model = np.loadtxt("../../tools/T_aurora_2_model", delimiter=",")
T_tip_2_coil = np.loadtxt("../../tools/T_tip_2_coil", delimiter=",")

motor_setpoints = continuum_arduino.load_motor_setpoints("../../tools/motor_setpoints")

theta = np.linspace(np.pi / (2 * theta_steps), np.pi / 2, theta_steps)
phi = np.arange(0, 2 * np.pi, np.pi / 2)

container = utils_data.DataContainer()
container.set_date_and_time()
container.prefix = "output/data"

l0 = 64
cable_deltas = 12.5 * np.ones((4, num_measurements))

for i in range(4):
    for j in range(theta_steps):
        sample = theta_steps * i + j

        kappa = theta[j] / l0
        seg_params = (l0, kappa, phi[i])

        cable_delta_list = mike_cc.one_seg_inverse_kinematics(seg_params, cable_positions)
        for k in range(samples_per_position):
            meas = sample * samples_per_position + k
            cable_deltas[i, meas] = cable_delta_list[i]

pos = np.nan * np.zeros((3, num_measurements))
tang = np.nan * np.zeros((3, num_measurements))

for i in range(4):
    print(f"Motor: {i}")
    continuum_arduino.write_motor_vals(arduino, motor_setpoints)
    time.sleep(4)

    for j in range(theta_steps):
        print(f"Theta: {180/np.pi*theta[j]:.2f}")
        sample = theta_steps * i + j
        dls = cable_deltas[:, sample * samples_per_position].flatten().tolist()

        motor_vals = continuum_arduino.one_seg_dl_2_motor_vals(dls, motor_setpoints)
        continuum_arduino.write_motor_vals(arduino, motor_vals)
        time.sleep(1)

        for k in range(samples_per_position):
            meas = samples_per_position * sample + k

            aurora_transform = {}
            counter = 0
            while not aurora_transform and counter < transform_attempts:
                aurora_transform = continuum_aurora.get_aurora_transforms(
                    aurora, probe_list
                )
                counter += 1

            # Convert raw aurora data to transformation matrix
            try:
                T = continuum_aurora.get_T_tip_2_model(
                    aurora_transform["0A"], T_aurora_2_model, T_tip_2_coil
                )

            except:
                aurora_transform = {}
                counter = 0
                while not aurora_transform and counter < transform_attempts:
                    aurora_transform = continuum_aurora.get_aurora_transforms(
                        aurora, probe_list
                    )
                    counter += 1

                T = continuum_aurora.get_T_tip_2_model(
                    aurora_transform["0A"], T_aurora_2_model, T_tip_2_coil
                )

            pos[:, meas] = T[0:3, 3]
            tang[:, meas] = kinematics.dcm_2_tang(T[0:3, 0:3])

        time.sleep(1)

continuum_arduino.write_motor_vals(arduino, motor_setpoints)

container.from_raw_data(container.date, container.time, 4, num_measurements, cable_deltas, pos, tang)
container.file_export()
