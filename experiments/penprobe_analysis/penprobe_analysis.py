#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import glob

penprobe_files = glob.glob("../../tools/penprobe*")

penprobe_positions = np.zeros((3, len(penprobe_files)))

for i in range(len(penprobe_files)):
    penprobe_positions[:, i] = np.loadtxt(penprobe_files[i], delimiter=",")

avg_penprobe = penprobe_positions.mean(axis=1)
penprobe_std = np.std(penprobe_positions, axis=1, ddof=1)
penprobe_error = np.linalg.norm(
    penprobe_positions - avg_penprobe.reshape((-1, 1)), axis=0
)
penprobe_rmse = np.sqrt((penprobe_error**2).mean())
print(penprobe_rmse)

filtered_penprobe_positions = penprobe_positions[:, penprobe_positions[1, :] < 1.5]
avg_filtered_penprobe = filtered_penprobe_positions.mean(axis=1)
filtered_penprobe_error = np.linalg.norm(
    filtered_penprobe_positions - avg_filtered_penprobe.reshape((-1, 1)), axis=0
)
filtered_penprobe_rmse = np.sqrt((filtered_penprobe_error**2).mean())
print(filtered_penprobe_rmse)

ax = plt.figure()
plt.plot(
    penprobe_positions[0, :],
    penprobe_positions[1, :],
    "o",
    label=f"All Penprobes (RMSE: {penprobe_rmse:.2f} mm)",
)
plt.plot(
    filtered_penprobe_positions[0, :],
    filtered_penprobe_positions[1, :],
    "x",
    label=f"Filtered Penprobes (RMSE: {filtered_penprobe_rmse:.2f} mm)",
)
plt.title(f"x-y Plane Projection of Pivot Calibrations")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.legend(loc="lower right")
plt.show()
