#!/bin/python3
import numpy as np
import serial
import time
import continuum_arduino
import continuum_aurora
import constant_curvature_utils
import mike_constant_curvature
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
rest_delay = 1

zero_meas_filename = "zero.csv"
meas_filename = "meas.csv"

thetas = np.linspace(np.pi / (2 * theta_steps), np.pi / 2, theta_steps)
phis = np.linspace(-np.pi / 2, 3 * np.pi / 2 - 2 * np.pi / phi_steps, phi_steps)

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
zero_meas = np.zeros(((1 + theta_steps * phi_steps) * samples_per_position, 8))
meas = np.zeros((phi_steps * theta_steps * samples_per_position, 15))

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

    zero_meas[i, 1] = i
    zero_meas[i, 2:5] = pos
    zero_meas[i, 5:8] = tang

starting_position = zero_meas[0:samples_per_position, 2:5].mean(axis=0)
starting_orientation = zero_meas[0:samples_per_position, 5:8].mean(axis=0)

T_start = np.identity(4)
T_start[0:3, 0:3] = kinematics.tang_2_dcm(starting_orientation)
T_start[0:3, 3] = starting_position

# Print out starting transform to make sure everything looks alright
print(T_start)

# Main data collection loop
for i in range(theta_steps):
    theta = thetas[i]
    print("theta:", np.rad2deg(theta))
    for j in range(phi_steps):
        sample = i * phi_steps + j
        phi = phis[j]

        print("phi:", np.rad2deg(phi))

        # Convert mike CC params (l, theta, phi) to webster CC params (l, kappa, phi)
        seg_params = constant_curvature_utils.mike_2_webster_params(64, theta, phi)

        # Using CC model, get cable displacements
        dls = mike_constant_curvature.one_seg_inverse_kinematics(
            seg_params, cable_positions
        )

        # Convert cable displacements to motor commands
        motor_vals = continuum_arduino.one_seg_dl_2_motor_vals(dls, motor_setpoints)

        # Write commands to motor and wait to get there
        continuum_arduino.write_motor_vals(arduino, motor_vals)
        time.sleep(rest_delay)

        # Sample tip state using
        for k in range(samples_per_position):
            row = sample * samples_per_position + k

            aurora_transform = {}
            counter = 0
            while not aurora_transform and counter < transform_attempts:
                aurora_transform = continuum_aurora.get_aurora_transforms(
                    aurora, probe_list
                )

            T = continuum_aurora.get_T_tip_2_model(
                aurora_transform["0A"], T_aurora_2_model, T_tip_2_coil
            )

            pos = T[0:3, 3]
            tang = kinematics.dcm_2_tang(T[0:3, 0:3])

            # Store data
            meas[row, 0] = sample
            meas[row, 1] = k
            meas[row, 2] = theta
            meas[row, 3] = phi
            meas[row, 4] = seg_params[1]  # kappa
            meas[row, 5:9] = np.array(dls)
            meas[row, 9:12] = pos
            meas[row, 12:15] = tang

        # Return to (0,0) point
        continuum_arduino.write_motor_vals(arduino, motor_setpoints)
        time.sleep(rest_delay)

        # Get measurements of (0,0) state
        for k in range(samples_per_position):
            sample = i * phi_steps + j + 1
            row = sample * samples_per_position + k

            # Get data from aurora
            aurora_transform = {}
            counter = 0
            while not aurora_transform and counter < transform_attempts:
                aurora_transform = continuum_aurora.get_aurora_transforms(
                    aurora, probe_list
                )

            # Convert aurora data to tip transform
            T = continuum_aurora.get_T_tip_2_model(
                aurora_transform["0A"], T_aurora_2_model, T_tip_2_coil
            )

            pos = T[0:3, 3]
            tang = kinematics.dcm_2_tang(T[0:3, 0:3])

            # Store data
            zero_meas[row, 0] = sample
            zero_meas[row, 1] = i
            zero_meas[row, 2:5] = pos
            zero_meas[row, 5:8] = tang

# Write data
np.savetxt("meas.csv", meas, delimiter=",")
np.savetxt("zero.csv", zero_meas, delimiter=",")

# Close serial connections
aurora.close()
arduino.close()
