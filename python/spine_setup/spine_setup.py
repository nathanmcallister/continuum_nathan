#!/bin/python3
import serial
import continuum_arduino

NUM_SEGMENTS = 1

SERVO_MIN = 80;
SERVO_MAX = 530;
SERVO_MID = int((SERVO_MIN + SERVO_MAX)/2);
TIGHTENING_FACTOR = 100

tight_motor_cmds = [SERVO_MID] * 4 * NUM_SEGMENTS
loose_motor_cmds = [SERVO_MID - TIGHTENING_FACTOR] * 4 * NUM_SEGMENTS

arduino = continuum_arduino.init_arduino()

continuum_arduino.write_motor_vals(arduino, tight_motor_cmds)

state = 0

while state >= 0:
    if state % 2 == 0:
        continuum_arduino.write_motor_vals(arduino, tight_motor_cmds)
        print("Motors tightened.")
    elif state % 2 == 1:
        continuum_arduino.write_motor_vals(arduino, loose_motor_cmds)
        print("Motors loosened.")

    i = input("Type s to switch state, e to exit: ")

    if i == 's':
        state += 1
    elif i == 'e':
        state = -1

arduino.close()
