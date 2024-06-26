#!/bin/python3
import numpy as np
import time
from continuum_aurora import ContinuumAurora

T_aurora_2_model = np.loadtxt("../../tools/T_aurora_2_model", delimiter=",")
T_tip_2_coil = np.loadtxt("../../tools/T_tip_2_coil", delimiter=",")

aurora = ContinuumAurora(T_aurora_2_model, T_tip_2_coil)

counter = 0
start = time.time_ns()
while counter < 40:
    transforms = aurora.read_aurora_transforms(["0A", "0B"])
    t = time.time_ns()
    print(
        counter,
        "|",
        (t - start) / 1e9,
        "\n",
        aurora.get_T_tip_2_model(transforms["0B"]),
    )
    counter += 1
    start = t
