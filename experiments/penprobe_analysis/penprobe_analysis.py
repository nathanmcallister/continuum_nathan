#!/bin/python3
import numpy as np
import matplotlib as plt
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
