#!/bin/python3
import serial
import continuum_arduino
import continuum_aurora

setpoints = [305, 305, 305, 305]
dls = [0, 0, 0, 0]
motor_vals = continuum_arduino.one_seg_dl_2_motor_values(dls, setpoints)
print(motor_vals)
arduino = serial.Serial("/dev/ttyACM0", 115200, timeout=1)
aurora = serial.Serial("/dev/ttyUSB0", 115200, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=1, xonxoff=False)
#print("Motor values sent:")
#print(continuum_arduino.write_motor_vals(motor_vals, arduino))

probe_list = ['0A', '0B']
continuum_aurora.request_aurora_packet(aurora, probe_list)
pkt = continuum_aurora.get_aurora_packet(aurora, 1)

with open("bytestr.txt", 'wb') as file:
    file.write(pkt)

file.close()

arduino.close()
aurora.close()
