#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from utils_data import DataContainer

container = DataContainer()
container.file_import("output/data_2024_07_03_14_14_04.dat")

cable_dls, pos, tang = container.to_numpy()

distances = np.zeros(25)

for i in range(25):
    distances[i] = np.linalg.norm(pos[:, 160 * i] - pos[:, 160 * (i + 1) - 1])

distances = sorted(
    list(enumerate(distances.tolist())), key=lambda x: x[1], reverse=True
)
discover_indices = False

if discover_indices:
    furthest_travel = [x[0] for x in distances]
    for idx in furthest_travel:
        travel_dist = np.linalg.norm(
            pos[:, 160 * idx : 160 * (idx + 1)]
            - pos[:, 160 * (idx + 1) - 1].reshape((3, 1)),
            axis=0,
        )
        plt.figure()
        plt.plot(np.arange(160) / 40, travel_dist)
        plt.title(f"Movement {idx}")
        plt.show()

decay_idx = 19
rapid_idx = 22

decay_pos = pos[:, 160 * decay_idx : 160 * (decay_idx + 1)]
rapid_pos = pos[:, 160 * rapid_idx : 160 * (rapid_idx + 1)]

decay_dist = np.linalg.norm(decay_pos - decay_pos[:, -1].reshape((3, 1)), axis=0)
rapid_dist = np.linalg.norm(rapid_pos - rapid_pos[:, -1].reshape((3, 1)), axis=0)

plt.figure()
plt.plot(np.arange(160) / 40, decay_dist, label="Loosening Trajectory")
plt.plot(np.arange(160) / 40, rapid_dist, label="Tightening Trajectory")
plt.title("Trajectory Distance from Endpoint vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Distance from Trajectory Endpoint (mm)")
plt.legend()

ax = plt.figure().add_subplot(projection="3d")
ax.plot(decay_pos[0, :], decay_pos[1, :], decay_pos[2, :], label="Loosening Trajectory")
ax.scatter(
    decay_pos[0, 0],
    decay_pos[1, 0],
    decay_pos[2, 0],
    color="C0",
    label="Loosening Trajectory Start",
)
ax.plot(
    rapid_pos[0, :], rapid_pos[1, :], rapid_pos[2, :], label="Tightening Trajectory"
)
ax.scatter(
    rapid_pos[0, 0],
    rapid_pos[1, 0],
    rapid_pos[2, 0],
    color="C1",
    label="Tightening Trajectory Start",
)
plt.title("3D View of Trajectories")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
ax.set_zlabel("z (mm)")
plt.legend()
plt.show()
