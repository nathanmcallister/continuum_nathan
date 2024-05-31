#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt

desired = np.loadtxt("output/trajectory.dat", delimiter=",")
pos = np.loadtxt("output/true_trajectory.dat", delimiter=",")
pos_bins = np.zeros((8, 3, 128))
for i in range(8):
    pos_bins[i, :, :] = pos[:, 128 * i : 128 * (i + 1)]

mean_pos = pos_bins.mean(axis=0)
repeatability = pos_bins - mean_pos
repeatability = np.linalg.norm(repeatability, axis=1)
repeatability_rms = np.sqrt((repeatability**2).mean())
print(repeatability_rms)

error = pos_bins - desired
error = np.linalg.norm(error, axis=1)
error_rms = np.sqrt((error**2).mean())
print(error_rms)

plt.figure()
plt.plot(mean_pos[0, :], mean_pos[1, :])

plt.figure()
plt.plot(desired[0, :], desired[1, :], "-o")
plt.plot(pos[0, :], pos[1, :], "-x")
plt.title(f"Open Loop Repeatability (RMSE: {repeatability_rms:.3f} mm)")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.legend(["Desired Trajectory", "Open Loop Trajectory"], loc="lower right")
plt.show()
