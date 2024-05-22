#!/bin/python3
import time
import serial
import numpy as np
import scipy.optimize as opt
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple
import ANN
import kinematics
import utils_data
import continuum_arduino
import continuum_aurora

trajectory = np.loadtxt("output/trajectory.dat", delimiter=",")
model_pos = np.loadtxt("output/model_trajectory.dat", delimiter=",")
open_loop_pos = np.loadtxt("output/true_trajectory.dat", delimiter=",")
closed_loop_pos = np.loadtxt("output/closed_loop_trajectory.dat", delimiter=",")

num_points = trajectory.shape[1]
closed_loop_steps = 10
model_error = np.linalg.norm(model_pos - trajectory, axis=0)
open_loop_error = np.linalg.norm(open_loop_pos - trajectory, axis=0)
closed_loop_error = np.linalg.norm(
    closed_loop_pos[:, closed_loop_steps - 1 :: closed_loop_steps] - trajectory, axis=0
)

model_rmse = np.sqrt((model_error**2).mean())
open_loop_rmse = np.sqrt((open_loop_error**2).mean())
closed_loop_rmse = np.sqrt((closed_loop_error**2).mean())

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

ax = plt.figure().add_subplot(projection="3d")
ax.plot(trajectory[0, :], trajectory[1, :], trajectory[2, :], "-o", label="Trajectory")
ax.plot(
    model_pos[0, :],
    model_pos[1, :],
    model_pos[2, :],
    "-x",
    label=f"Model (RMSE: {model_rmse:.3f} mm)",
)
ax.plot(
    open_loop_pos[0, :],
    open_loop_pos[1, :],
    open_loop_pos[2, :],
    "-+",
    label=f"Open Loop (RMSE: {open_loop_rmse:.3f} mm)",
)
ax.plot(
    closed_loop_pos[0, closed_loop_steps - 1 :: closed_loop_steps],
    closed_loop_pos[1, closed_loop_steps - 1 :: closed_loop_steps],
    closed_loop_pos[2, closed_loop_steps - 1 :: closed_loop_steps],
    "-2",
    label=f"Closed Loop (RMSE: {closed_loop_rmse:.3f} mm)",
    color=colors[-1],
)
plt.xlim((-32, 32))
plt.ylim((-32, 32))
ax.set_zlim((0, 64))
plt.title("Closed Loop Trajectory Tracking")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
ax.set_zlabel("z (mm)")
plt.legend()

plt.figure()
plt.plot(trajectory[0, :], trajectory[1, :], "o", label="Trajectory")
plt.plot(
    model_pos[0, :],
    model_pos[1, :],
    "x",
    label=f"Model (RMSE: {model_rmse:.3f} mm)",
)
plt.plot(
    open_loop_pos[0, :],
    open_loop_pos[1, :],
    "+",
    label=f"Open Loop (RMSE: {open_loop_rmse:.3f} mm)",
)
plt.plot(
    closed_loop_pos[0, closed_loop_steps - 1 :: closed_loop_steps],
    closed_loop_pos[1, closed_loop_steps - 1 :: closed_loop_steps],
    "2",
    label=f"Closed Loop (RMSE: {closed_loop_rmse:.3f} mm)",
    color=colors[-1],
)

plt.title("Closed Loop Trajectory Tracking")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.legend(loc="lower right")

plt.figure()
plt.plot(
    closed_loop_pos[0, 80 * closed_loop_steps : (80 + 1) * closed_loop_steps]
    - trajectory[0, 80],
    label="$e_x$",
)
plt.plot(
    closed_loop_pos[1, 80 * closed_loop_steps : (80 + 1) * closed_loop_steps]
    - trajectory[1, 80],
    label="$e_y$",
)
plt.plot(
    closed_loop_pos[2, 80 * closed_loop_steps : (80 + 1) * closed_loop_steps]
    - trajectory[2, 80],
    label="$e_z$",
)
plt.plot(
    np.linalg.norm(
        closed_loop_pos[:, 80 * closed_loop_steps : (80 + 1) * closed_loop_steps]
        - trajectory[:, 80].reshape((-1, 1)),
        axis=0,
    ),
    label="$|e|$",
)
plt.legend()
plt.title("Closed Loop Error vs Step")
plt.xlabel("Step")
plt.ylabel("Position Error (mm)")

plt.figure()
plt.plot(model_error, label="Model", color=colors[1])
plt.plot(open_loop_error, label="Open Loop", color=colors[2])
plt.plot(closed_loop_error, color=colors[-1], label="Closed Loop")
plt.title("Position Error Along Trajectory")
plt.xlabel("Step")
plt.ylabel("Error (mm)")
plt.legend(loc="upper left")
plt.show()
