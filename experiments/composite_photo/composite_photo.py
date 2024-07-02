#!/bin/python3
import numpy as np
import time
from continuum_arduino import ContinuumArduino

arduino = ContinuumArduino()
wait_time = 2

for i in range(16):
    dls = np.array([-i, 0.0, 0.0, 0.0], dtype=float)
    arduino.write_dls(dls)
    time.sleep(wait_time)

arduino.write_dls(np.zeros(4, dtype=float))
