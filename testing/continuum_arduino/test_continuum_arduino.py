#!/bin/python3
import os
import pty
import serial
import time
from multiprocessing import Process
import numpy as np
from continuum_arduino import ContinuumArduino, crc_add_bytes


class colors:
    """Colors class:reset all colors with colors.reset; two
    sub classes fg for foreground
    and bg for background; use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.greenalso, the generic bold, disable,
    underline, reverse, strike through,
    and invisible work with the main class i.e. colors.bold"""

    reset = "\033[0m"
    bold = "\033[01m"
    disable = "\033[02m"
    underline = "\033[04m"
    reverse = "\033[07m"
    strikethrough = "\033[09m"
    invisible = "\033[08m"

    class fg:
        black = "\033[30m"
        red = "\033[31m"
        green = "\033[32m"
        orange = "\033[33m"
        blue = "\033[34m"
        purple = "\033[35m"
        cyan = "\033[36m"
        lightgrey = "\033[37m"
        darkgrey = "\033[90m"
        lightred = "\033[91m"
        lightgreen = "\033[92m"
        yellow = "\033[93m"
        lightblue = "\033[94m"
        pink = "\033[95m"
        lightcyan = "\033[96m"

    class bg:
        black = "\033[40m"
        red = "\033[41m"
        green = "\033[42m"
        orange = "\033[43m"
        blue = "\033[44m"
        purple = "\033[45m"
        cyan = "\033[46m"
        lightgrey = "\033[47m"


def arduino_tester(slave_name):
    arduino = serial.Serial(slave_name, 115200, timeout=1)

    while True:
        # Read until start sequence is found
        bytes_until_start = arduino.read_until(bytearray([0x10, 0x02]))

        if len(bytes_until_start) < 2 or not (
            bytes_until_start[-2] == 0x10 and bytes_until_start[-1] == 0x02
        ):
            continue

        # Read until end sequence is found
        bytes_until_end = arduino.read_until(bytearray([0x10, 0x03]))
        if not (bytes_until_end[-2] == 0x10 and bytes_until_end[-1] == 0x03):
            continue
        # Extract payload and crc
        payload = bytes_until_end[:-3]
        crc = bytes_until_end[-3]

        # Process the payload
        # For simplicity, we are always sending an ACK
        ack_payload = bytearray([0x05, 0x02, 0x10, 0x06])
        ack_crc = crc_add_bytes(0, ack_payload)
        ack_packet = bytearray([0x10, 0x02])
        ack_packet += ack_payload
        ack_packet += bytearray([ack_crc, 0x10, 0x03])
        arduino.write(ack_packet)


# Start the mock Arduino process
def start_arduino_tester(slave_name):
    p = Process(target=arduino_tester, args=(slave_name,))
    p.start()
    return p


if __name__ == "__main__":

    arduino_tester_process = start_arduino_tester("/dev/pts/5")
    arduino = ContinuumArduino(serial_port_name="/dev/pts/6", testing=True)
    time.sleep(5)

    # Receive update_num_motors, send back

    arduino_tester_process.terminate()
    arduino_tester_process.join()
