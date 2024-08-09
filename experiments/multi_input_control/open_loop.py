#!/bin/python3
import numpy as np
from time import sleep
from datetime import datetime
from pathlib import Path
from multi_input import MultiInputModel
from continuum_arduino import ContinuumArduino
from continuum_aurora import ContinuumAurora
from continuum_control import PositionController
from kinematics import dcm_2_tang
from utils_data import DataContainer

"""
open_loop.py
Author: Cameron Wolfe
Created: 08/01/2024

Tracks a trajectory in open loop using a MultiInputModel and a
MultiInputPositionController.
"""

# Parameters
wait_time = 4  # seconds

# Trajectory
trajectory = np.loadtxt(Path("trajectories/trajectory.dat"), delimiter=",")
num_points = trajectory.shape[1]

# Continuum setup
T_aurora_2_model = np.loadtxt(Path("../../tools/T_aurora_2_model"), delimiter=",")
T_tip_2_coil = np.loadtxt(Path("../../tools/T_tip_2_coil"), delimiter=",")

arduino = ContinuumArduino()
aurora = ContinuumAurora(T_aurora_2_model, T_tip_2_coil)

# Load model (Make sure hidden layer size is correct)
model = MultiInputModel(4, 1, 1, [64, 32])
model.load("./models/big_07_30_2024/2024_07_30_21_08_15.pt")

# Setup controller
controller = PositionController(model, 1, 0.1)

# Data allocation
cable_deltas = np.zeros((4, num_points))
pos = np.zeros((3, num_points))
tang = np.zeros((3, num_points))
now = datetime.now()
container = DataContainer()
container.prefix = "output/open_loop"

# Data collection
for i in range(num_points):
    # Get cable_deltas for trajectory point
    cable_deltas[:, i] = controller.open_loop_step(trajectory[:, i])

    # Move robot
    arduino.write_dls(cable_deltas[:, i])
    sleep(wait_time)

    # Get tip transform
    transforms = aurora.get_aurora_transforms(["0A"])
    T_tip_2_model = aurora.get_T_tip_2_model(transforms["0A"])

    # Extract data
    pos[:, i] = T_tip_2_model[:3, 3]
    tang[:, i] = dcm_2_tang(T_tip_2_model[:3, :3])

# Data packaging and export
container.from_raw_data(
    (now.year, now.month, now.day),
    (now.hour, now.minute, now.second),
    4,
    num_points,
    cable_deltas,
    pos,
    tang,
)
container.file_export()
