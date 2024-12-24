#!/bin/python3
import serial
import time
import continuum_arduino

NUM_SEGMENTS = 1

Fs = 324
Ts = 1 / Fs
num_steps = 2**12
futaba_cw60 = 920
futaba_ccw60 = 2120

us2s = 10 ** (-6)
us2steps = num_steps / Ts * us2s

cw60 = futaba_cw60 * us2steps
ccw60 = futaba_ccw60 * us2steps

servo_mid = int((cw60 + ccw60) / 2)
servo_pos = servo_mid

cmd2deg = 120 / (ccw60 - cw60)
print(ccw60, servo_mid, cw60, cmd2deg)

arduino = continuum_arduino.init_arduino()


while True:
    print(f"Servo position: {servo_pos}")
    print(f"Angle: {(servo_pos - servo_mid) * cmd2deg:.2f} degrees")
    motor_cmds = [servo_pos] * 4 * NUM_SEGMENTS
    continuum_arduino.write_motor_vals(arduino, motor_cmds)

    i = input("[C]lockwise, [A]nticlockwise, or [E]xit? ").lower()

    if i[0] == "c":
        delta = 10
        if i[1:]:
            try:
                delta = max(0, min(200, int(float(i[1:]))))
            except:
                print("Invalid step size")
        servo_pos -= delta

    elif i[0] == "a":
        delta = 10
        if i[1:]:
            try:
                delta = max(0, min(200, int(float(i[1:]))))
            except:
                print("Invalid step size")
        servo_pos += delta

    elif i[0] == "e":
        break


arduino.close()
