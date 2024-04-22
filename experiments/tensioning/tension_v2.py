#!/bin/python3
import numpy as np
import serial
import time
import continuum_arduino
import continuum_aurora
import kinematics

""" Tensioning
Cameron Wolfe 04/20/24

Tensions cables via incrementally tensioning them, and measuring displacement
from starting position.  If displacement exceeds a threshold, then the cables
are considered tensioned.  Repeats for each cable.
"""

# Parameters
no_movement_steps = 2 # Number of steps of no movement before cable is loose
no_movement_threshold = 0.2 # If the position changes by less than 0.2 for the given number of steps, then you're tensioned
max_delta = 15  # mm - Cannot pull a cable more than 15 mm from its starting length
step_size = 0.1 # mm - How many mm we pull the cable each step
loose_delta = 5
tight_delta = -5
num_motors = 4
servo_min = 80
servo_max = 530

# Load tools and setpoints
T_aurora_2_model = np.loadtxt("../../tools/T_aurora_2_model", delimiter=",")
T_tip_2_coil = np.loadtxt("../../tools/T_tip_2_coil", delimiter=",")

motor_setpoints = [int((servo_min + servo_max) / 2)] * num_motors

# Assuming that both pen and probe are plugged in, but it will work if just the probe is plugged in
probe_list = ["0A", "0B"]

# Open serial connections to arduino and aurora
arduino = continuum_arduino.init_arduino()
aurora = continuum_aurora.init_aurora()

# Go to setpoints
continuum_arduino.write_motor_vals(arduino, motor_setpoints)
time.sleep(0.5)

# Loosen cables
loose_length = [loose_delta] * num_motors
starting_motor_vals = continuum_arduino.one_seg_dl_2_motor_vals(
    loose_length, motor_setpoints
)
continuum_arduino.write_motor_vals(arduino, starting_motor_vals)
time.sleep(0.5)

# Get initial transform from aurora
starting_transforms = continuum_aurora.get_aurora_transforms(aurora, probe_list)

# Determine if both pen and coil are plugged in or not
coil = "0A"
if len(starting_transforms.keys()) == 2:
    coil = "0B"

# Prepare for tensioning
positions = np.zeros((num_motors, 3, int((loose_delta - tight_delta) / step_size) + 1))
final_deltas = [0.0] * num_motors

# Loop through each motor
for i in range(num_motors):

    # Go to loose position
    continuum_arduino.write_motor_vals(arduino, starting_motor_vals)
    time.sleep(2)

    positions[i, :] = np.nan;

    delta = tight_delta
    j = 0
    tensioned = False
    while delta < loose_delta and not tensioned:

        deltas = loose_length[:]
        deltas[i] = delta

        new_motor_vals = continuum_arduino.one_seg_dl_2_motor_vals(deltas, motor_setpoints)
        continuum_arduino.write_motor_vals(arduino, new_motor_vals)
        if j == 0:
            time.sleep(2)
        else:
            time.sleep(1)

        transforms = continuum_aurora.get_aurora_transforms(aurora, probe_list)

        T_tip_2_model = continuum_aurora.get_T_tip_2_model(
            transforms[coil], T_aurora_2_model, T_tip_2_coil
        )
        positions[i, :, j] = T_tip_2_model[0:3, 3]

        if j >= 5 * no_movement_steps:
            position_deltas = positions[i, :, j].reshape((3,1)) - positions[i, :, j-no_movement_steps:j]

            norm_position_deltas = np.linalg.norm(position_deltas, axis=0)
            print(f"{delta:.2f}:", positions[i, :, j], norm_position_deltas)

            if (norm_position_deltas < no_movement_threshold).all():
                tensioned = True
            else:
                delta += step_size
                j += 1
        else:
            print(f"{delta:.2f}:", positions[i, :, j])
            delta += step_size
            j += 1

    if not tensioned:
        print(f"Motor {i} did not tension properly, ran past bound");

    final_deltas[i] = delta
    continuum_arduino.write_motor_vals(arduino, starting_motor_vals)
    time.sleep(1)


# Reset to loose position
continuum_arduino.write_motor_vals(arduino, starting_motor_vals)
time.sleep(1)

# Using deltas, calculate new "tensioned" motor setpoints
final_motor_vals = continuum_arduino.one_seg_dl_2_motor_vals(
    final_deltas, motor_setpoints
)

# Go to the new setpoints
continuum_arduino.write_motor_vals(arduino, final_motor_vals)

# Output setpoints
file = open("../../tools/motor_setpoints", "w")
for i in range(4):
    if i < 3:
        file.write(str(final_motor_vals[i]) + ",")
    else:
        file.write(str(final_motor_vals[i]))
file.close()

# Output displacement data
for i in range(num_motors):
    np.savetxt(f"output/motor_{i}_positions.dat", positions[i, :, :], delimiter=",")

# Get one final transform and display it
transforms = continuum_aurora.get_aurora_transforms(aurora, probe_list)
T_tip_2_model = continuum_aurora.get_T_tip_2_model(
    transforms[coil], T_aurora_2_model, T_tip_2_coil
)
print(T_tip_2_model)

# Close serial connections
arduino.close()
aurora.close()
