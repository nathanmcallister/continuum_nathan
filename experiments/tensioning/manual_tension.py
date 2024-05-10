#!/bin/python3
import numpy as np
import serial
import time
import continuum_arduino
import continuum_aurora
import kinematics

num_motors = 4
servo_min = 1221
servo_max = 2813

step_size = 0.1

loose_length = 1

arduino = continuum_arduino.init_arduino()

user_input = input("Would you like to start with existing setpoints? [Y/n] ")

motor_setpoints = [] * num_motors
deltas = [0] * num_motors

if user_input == "n":
    motor_setpoints = [int((servo_max + servo_min) / 2)] * num_motors
else:
    motor_setpoints = continuum_arduino.load_motor_setpoints(
        "../../tools/motor_setpoints"
    )

current_setpoints = continuum_arduino.one_seg_dl_2_motor_vals(deltas, motor_setpoints)

continuum_arduino.write_motor_vals(arduino, current_setpoints)

state = 0
current_motor = 0

while state >= 0:
    current_setpoints = continuum_arduino.one_seg_dl_2_motor_vals(
        deltas, motor_setpoints
    )
    continuum_arduino.write_motor_vals(arduino, current_setpoints)

    if state == 0:
        user_input = input(
            f"Input the motor you would like to tension (0:{num_motors-1}) or (e)xit "
        )

        if user_input.isdigit():
            selected_motor = int(user_input)
            if 0 <= selected_motor < num_motors:
                current_motor = selected_motor
                state = 1
        elif user_input == "e" or user_input == "exit":
            state = -1
        else:
            print("Invalid input: " + user_input)

    elif state == 1:
        user_input = input("(T)ighten, (l)oosen, or (e)xit? ")

        if user_input == "t" or user_input == "tighten":
            deltas[current_motor] -= step_size

        elif user_input == "l" or user_input == "loosen":
            deltas[current_motor] += step_size

        elif user_input == "e" or user_input == "exit":
            state = 0

        else:
            print("Invalid input: " + user_input)

print("Deltas:", deltas)
print("Current setpoints:", current_setpoints)

user_input = input("Would you like to save? [y/N] ")

if user_input == "y":
    with open("../../tools/motor_setpoints", "w") as f:
        for i in range(num_motors):
            if i != num_motors - 1:
                f.write(f"{current_setpoints[i]},")
            else:
                f.write(f"{current_setpoints[i]}")
