#!/bin/python3
import serial
import numpy as np
from continuum_arduino import ContinuumArduino

arduino = ContinuumArduino()
arduino.write_dls(np.zeros(4))

state = 0

while state >= 0:
    if state % 2 == 0:
        arduino.write_dls(np.zeros(4))
        print("Motors tightened.")
    elif state % 2 == 1:
        arduino.write_dls(np.full(4, 12))
        print("Motors loosened.")

    i = input("Type s to switch state, e to exit: ")

    if i == "s":
        state += 1
    elif i == "e":
        state = -1
