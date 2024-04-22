#!/bin/python3
import numpy as np
import serial
import time
import continuum_arduino
import continuum_aurora
import utils_cc
import utils_data
import mike_cc
import kinematics

""" Constant Curvature
Cameron Wolfe 04/09/24

Sweeps through a range of thetas (tilt angle) and phis (sweep angle), uses CC
model to generate cable length displacements, and then moves the spine to
positions.  Measures position based on these commands, and repeats for all
points.
"""

# Parameters
samples_per_position = 5
transform_attempts = 5
theta_steps = 12
phi_steps = 24
rest_delay = 0.5
num_motors = 4

num_measurements = theta_steps * phi_steps * samples_per_position
num_zero_measurements = num_measurements + samples_per_position

thetas = np.linspace(np.pi / (2 * theta_steps), np.pi / 2, theta_steps)
phis = np.linspace(-np.pi , np.pi - 2 * np.pi / phi_steps, phi_steps)

# Load tools and setpoints
T_aurora_2_model = np.loadtxt("../../tools/T_aurora_2_model", delimiter=",")
T_tip_2_coil = np.loadtxt("../../tools/T_tip_2_coil", delimiter=",")

motor_setpoints = continuum_arduino.load_motor_setpoints(
    "../../tools/motor_setpoints"
)  # Delta length = 0 positions

# Setup cable positions
cable_positions = [(8, 0), (0, 8), (-8, 0), (0, -8)]

# Initialize Aurora and Arduino
aurora = continuum_aurora.init_aurora()
probe_list = ["0A"]
arduino = continuum_arduino.init_arduino()

# Move spine to (0,0)
continuum_arduino.write_motor_vals(arduino, motor_setpoints)
time.sleep(0.5)

# Initialize data matrices
meas_cable_deltas = np.nan * np.zeros((num_motors, num_measurements))
meas_pos = np.nan * np.zeros((3, num_measurements))
meas_tang = np.nan * np.zeros((3, num_measurements))

zero_cable_deltas = np.zeros((num_motors, num_zero_measurements))
zero_pos = np.nan * np.zeros((3, num_zero_measurements))
zero_tang = np.nan * np.zeros((3, num_zero_measurements))

# Create empty data_containers
zero_container = utils_data.DataContainer()
zero_container.set_date_and_time()
zero_container.prefix = "output/zero"
meas_container = utils_data.DataContainer()
meas_container.set_date_and_time()
meas_container.prefix = "output/meas"

# Get starting position
for i in range(samples_per_position):
    aurora_transform = {}
    counter = 0
    while not aurora_transform and counter < transform_attempts:
        aurora_transform = continuum_aurora.get_aurora_transforms(aurora, probe_list)

    T = continuum_aurora.get_T_tip_2_model(
        aurora_transform["0A"], T_aurora_2_model, T_tip_2_coil
    )

    pos = T[0:3, 3]
    tang = kinematics.dcm_2_tang(T[0:3, 0:3])

    zero_pos[:, i] = pos
    zero_tang[:, i] = tang

starting_position = zero_pos[:, 0:samples_per_position].mean(axis=1)
starting_orientation = zero_tang[:, 0:samples_per_position].mean(axis=1)

T_start = np.identity(4)
T_start[0:3, 0:3] = kinematics.tang_2_dcm(starting_orientation)
T_start[0:3, 3] = starting_position

# Print out starting transform to make sure everything looks alright
print(T_start)

# Main data collection loop
for i in range(phi_steps):
    phi = phis[i]
    print("phi: {:.2f}".format(np.rad2deg(phi)))
    for j in range(theta_steps):
        theta = thetas[j]
        print("theta: {:.2f}".format(np.rad2deg(theta)))
        sample = i * theta_steps + j

        # Convert mike CC params (l, theta, phi) to webster CC params (l, kappa, phi)
        seg_params = utils_cc.mike_2_webster_params(64, theta, phi)

        # Using CC model, get cable displacements
        dls = mike_cc.one_seg_inverse_kinematics(
            seg_params, cable_positions
        )

        # Convert cable displacements to motor commands
        motor_vals = continuum_arduino.one_seg_dl_2_motor_vals(dls, motor_setpoints)

        # Write commands to motor and wait to get there
        time.sleep(rest_delay)
        continuum_arduino.write_motor_vals(arduino, motor_vals)
        time.sleep(rest_delay)

        # Sample tip state using
        for k in range(samples_per_position):
            meas = sample * samples_per_position + k

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


            pos = T[0:3, 3]
            tang = kinematics.dcm_2_tang(T[0:3, 0:3])

            # Store data
            meas_cable_deltas[:, meas] = np.array(dls)
            meas_pos[:, meas] = pos
            meas_tang[:, meas] = tang

        # Return to (0,0) point
        time.sleep(rest_delay)
        continuum_arduino.write_motor_vals(arduino, motor_setpoints)
        time.sleep(3*rest_delay)

        # Get measurements of (0,0) state
        sample += 1
        for k in range(samples_per_position):
            meas = sample * samples_per_position + k

            # Get data from aurora
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

            pos = T[0:3, 3]
            tang = kinematics.dcm_2_tang(T[0:3, 0:3])

            # Store data
            zero_pos[:, meas] = pos
            zero_tang[:, meas] = tang

# Fill data containers
zero_container.from_raw_data(zero_container.date, zero_container.time, num_motors, num_zero_measurements, zero_cable_deltas, zero_pos, zero_tang)
meas_container.from_raw_data(meas_container.date, meas_container.time, num_motors, num_measurements, meas_cable_deltas, meas_pos, meas_tang)

# Write data
zero_container.file_export()
meas_container.file_export()

# Close serial connections
aurora.close()
arduino.close()
