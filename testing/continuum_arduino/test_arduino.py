#!/bin/python3
import numpy as np
from continuum_arduino import ContinuumArduino

arduino = ContinuumArduino(testing=True)

# Run tests
# Initialization
uninit = not arduino.write_dls(np.array([0, 1, 2, 3.0]), attempts=1)

# update_num_motors
dim_mismatch = not arduino.update_num_motors(
    0, np.array([1]), np.array([15]), attempts=1
)
invalid_num_motors = not arduino.update_num_motors(
    17, np.ones(17), np.ones(17), attempts=1
)
valid_num_motors = arduino.update_num_motors(
    4, arduino.motor_setpoints, arduino.wheel_radii, attempts=1
)

# update_oscillator_frequency
invalid_o_freq = not arduino.update_oscillator_frequency(1000, attempts=1)
valid_o_freq = arduino.update_oscillator_frequency(25_000, attempts=1)

# update_servo_frequency
invalid_s_freq = not arduino.update_servo_frequency(10000, attempts=1)
valid_s_freq = arduino.update_servo_frequency(300, attempts=1)

# write_dls
invalid_dls = not arduino.write_dls(np.ones(6), attempts=1)
valid_dls = arduino.write_dls(np.ones(4), attempts=1)

# Print results
print("Initialization test:")
print("Catches uninitialization:", uninit)
print("---\n")

print("update_num_motors tests:")
print("Handles a mismatch between dimensions of parameters:", dim_mismatch)
print("Handles an invalid number of motors:", invalid_num_motors)
print("Updates number of motors correctly:", valid_num_motors)
print("---\n")

print("update_oscillator_frequency tests:")
print("Handles invalid frequency:", invalid_o_freq)
print("Updates frequency correctly:", valid_o_freq)
print("---\n")

print("update_servo_frequency tests:")
print("Handles invalid frequency:", invalid_s_freq)
print("Updates frequency correctly:", valid_s_freq)
print("---\n")

print("write_dls tests:")
print("Handles incorrect dimension dls:", invalid_dls)
print("Sends dls correctly:", valid_dls)
