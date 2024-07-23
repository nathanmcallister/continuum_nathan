#!/bin/python3
import time
from pathlib import Path
import numpy as np
from continuum_arduino import ContinuumArduino
from continuum_aurora import ContinuumAurora
from mike_cc import MikeModel
import utils_cc
from utils_data import DataContainer
from kinematics import dcm_2_tang, tang_2_dcm
import kinematics

""" 
constant_curvature.py
Created: Cameron Wolfe 04/09/24
Updated: Cameron Wolfe 07/17/24

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
rest_delay = 4
num_motors = 4
sweep_phi = False

num_measurements = theta_steps * phi_steps * samples_per_position
num_zero_measurements = num_measurements + samples_per_position

thetas = np.linspace(np.pi / (2 * theta_steps), np.pi / 2, theta_steps)
phis = np.linspace(-np.pi, np.pi - 2 * np.pi / phi_steps, phi_steps)

# Load tools and setpoints
T_aurora_2_model = np.loadtxt(Path("../../tools/T_aurora_2_model"), delimiter=",")
T_tip_2_coil = np.loadtxt(Path("../../tools/T_tip_2_coil"), delimiter=",")

# Setup CC model
num_cables = 4
cable_positions = [(4, 0), (0, 4), (-4, 0), (0, -4)]
segment_length = 64
cc_model = MikeModel(num_cables, cable_positions, segment_length)

# Initialize Aurora and Arduino
probe_list = ["0A"]
aurora = ContinuumAurora(T_aurora_2_model, T_tip_2_coil)
arduino = ContinuumArduino()

# Move spine to (0,0)
arduino.write_dls(np.zeros(4))

# Initialize data matrices
meas_cable_deltas = np.nan * np.zeros((num_motors, num_measurements))
meas_pos = np.nan * np.zeros((3, num_measurements))
meas_tang = np.nan * np.zeros((3, num_measurements))

zero_cable_deltas = np.zeros((num_motors, num_zero_measurements))
zero_pos = np.nan * np.zeros((3, num_zero_measurements))
zero_tang = np.nan * np.zeros((3, num_zero_measurements))

# Create empty data_containers
zero_container = DataContainer()
zero_container.set_date_and_time()
zero_container.prefix = "output/zero"
meas_container = DataContainer()
meas_container.set_date_and_time()
meas_container.prefix = "output/meas"

# Get starting position
for i in range(samples_per_position):
    aurora_transform = {}
    counter = 0
    while not aurora_transform and counter < transform_attempts:
        aurora_transform = aurora.get_aurora_transforms(probe_list)

    T = aurora.get_T_tip_2_model(aurora_transform["0A"])

    pos = T[0:3, 3]
    tang = dcm_2_tang(T[0:3, 0:3])

    zero_pos[:, i] = pos
    zero_tang[:, i] = tang

starting_position = zero_pos[:, 0:samples_per_position].mean(axis=1)
starting_orientation = zero_tang[:, 0:samples_per_position].mean(axis=1)

T_start = np.identity(4)
T_start[0:3, 0:3] = tang_2_dcm(starting_orientation)
T_start[0:3, 3] = starting_position

# Print out starting transform to make sure everything looks alright
print(T_start)

if sweep_phi:
    # Main data collection loop
    for i in range(theta_steps):
        theta = thetas[i]
        print("theta: {:.2f}".format(np.rad2deg(theta)))
        for j in range(phi_steps):
            phi = phis[j]
            print("phi: {:.2f}".format(np.rad2deg(phi)))
            sample = i * phi_steps + j

            # Using CC model, get cable displacements
            dls = cc_model.inverse(np.array([theta, phi]))

            arduino.write_dls(dls)
            time.sleep(rest_delay)

            # Sample tip state using
            for k in range(samples_per_position):
                meas = sample * samples_per_position + k

                counter = 0
                while counter < transform_attempts:
                    try:
                        aurora_transform = aurora.get_aurora_transforms(probe_list)

                        T = aurora.get_T_tip_2_model(aurora_transform["0A"])
                        break

                    except:
                        counter += 1

                pos = T[0:3, 3]
                tang = dcm_2_tang(T[0:3, 0:3])

                # Store data
                meas_cable_deltas[:, meas] = np.array(dls)
                meas_pos[:, meas] = pos
                meas_tang[:, meas] = tang

            # Return to (0,0) point
            arduino.write_dls(np.zeros(4))
            time.sleep(rest_delay)

            # Get measurements of (0,0) state
            sample += 1
            for k in range(samples_per_position):
                meas = sample * samples_per_position + k

                # Get data from aurora
                counter = 0
                collecting
                while counter < transform_attempts:
                    try:
                        aurora_transform = aurora.get_aurora_transforms(probe_list)

                        T = aurora.get_T_tip_2_model(aurora_transform["0A"])
                        break

                    except:
                        counter += 1

                pos = T[0:3, 3]
                tang = kinematics.dcm_2_tang(T[0:3, 0:3])

                # Store data
                zero_pos[:, meas] = pos
                zero_tang[:, meas] = tang
else:
    # Main data collection loop
    for i in range(phi_steps):
        phi = phis[i]
        print("phi: {:.2f}".format(np.rad2deg(phi)))
        for j in range(theta_steps):
            theta = thetas[j]
            print("theta: {:.2f}".format(np.rad2deg(theta)))
            sample = i * theta_steps + j

            # Using CC model, get cable displacements
            dls = cc_model.inverse(np.array([theta, phi]))

            # Write cable displacements and wait
            arduino.write_dls(dls)
            time.sleep(rest_delay)

            # Sample tip state
            for k in range(samples_per_position):
                meas = sample * samples_per_position + k

                counter = 0
                while counter < transform_attempts:
                    try:
                        aurora_transform = aurora.get_aurora_transforms(probe_list)

                        T = aurora.get_T_tip_2_model(aurora_transform["0A"])
                        break

                    except:
                        counter += 1

                pos = T[0:3, 3]
                tang = dcm_2_tang(T[0:3, 0:3])

                # Store data
                meas_cable_deltas[:, meas] = np.array(dls)
                meas_pos[:, meas] = pos
                meas_tang[:, meas] = tang

            # Return to (0,0) point
            arduino.write_dls(np.zeros(4))
            time.sleep(rest_delay)

            # Get measurements of (0,0) state
            sample += 1
            for k in range(samples_per_position):
                meas = sample * samples_per_position + k

                # Get data from aurora
                counter = 0
                while counter < transform_attempts:
                    try:
                        aurora_transform = aurora.get_aurora_transforms(probe_list)

                        T = aurora.get_T_tip_2_model(aurora_transform["0A"])
                        break

                    except:
                        counter += 1

                pos = T[0:3, 3]
                tang = kinematics.dcm_2_tang(T[0:3, 0:3])

                # Store data
                zero_pos[:, meas] = pos
                zero_tang[:, meas] = tang

# Fill data containers
zero_container.from_raw_data(
    zero_container.date,
    zero_container.time,
    num_motors,
    num_zero_measurements,
    zero_cable_deltas,
    zero_pos,
    zero_tang,
)
meas_container.from_raw_data(
    meas_container.date,
    meas_container.time,
    num_motors,
    num_measurements,
    meas_cable_deltas,
    meas_pos,
    meas_tang,
)

# Write data
zero_container.file_export()
meas_container.file_export()
