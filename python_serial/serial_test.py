#!/bin/python3
import serial
import continuum_arduino
import continuum_aurora

setpoints = [305, 305, 305, 305]
dls = [0, 0, 0, 0]
motor_vals = continuum_arduino.one_seg_dl_2_motor_values(dls, setpoints)
print(motor_vals)

with serial.Serial("/dev/ttyACM0", 115200, timeout=1) as arduino:
    print("Motor values sent:")
    print(continuum_arduino.write_motor_vals(motor_vals, arduino))
    arduino.close()
