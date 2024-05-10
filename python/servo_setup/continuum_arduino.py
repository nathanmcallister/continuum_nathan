import serial
import struct
import math
import time
from typing import List


def init_arduino():
    return serial.Serial("/dev/ttyACM0", 115200, timeout=1)


def write_motor_vals(
    device: serial.Serial, motor_vals: List[int], error_delay: float = 0.5
) -> bool:
    # Important bytes for transmission
    DLE = 0x10
    STX = 0x02
    ETX = 0x03
    ACK = 0x06
    ERR = 0x15

    # Ensure correct baud rate
    if device.baudrate != 115200:
        print("Incorrect baud rate: " + str(device.baudrate))
        return False

    num_motors = len(motor_vals)

    # Form bytearray to send to arduino
    packet = bytearray()
    packet.append(DLE)
    packet.append(STX)
    packet.extend(struct.pack("B", 2 * num_motors))
    for val in motor_vals:
        packed_val = struct.pack("H", val)

        dle_idx = packed_val.find(DLE)
        # DLE in bytes, stuff it
        if dle_idx == 0:
            packet.append(DLE)
            packet.extend(packed_val)
        elif dle_idx == 1:
            packed.extend(packed_val)
            packet.append(DLE)
        # No DLE in bytes
        else:
            packet.extend(packed_val)

    packet.append(DLE)
    packet.append(ETX)

    # Try transmitting byte array up to 5 times
    success = False
    errors = 0
    while not success and errors < 5:
        device.write(packet)

        output = device.read(7)

        if len(output) == 7:
            if output[4] == ACK:
                success = True
            elif output[4] == ERR:
                print("ERR received")
                time.sleep(error_delay)
                errors += 1
            else:
                print("Neither ACK or ERR received")
                time.sleep(error_delay)
                errors += 1

        else:
            if len(output) == 0:
                print("Timeout")
                time.sleep(error_delay)
                errors += 1
            else:
                print("Transmission error, received incorrect length response")
                time.sleep(error_delay)
                errors += 1

    return success


def one_seg_dl_2_motor_vals(dls: List[float], setpoints: List[int]) -> List[int]:

    assert len(dls) == len(setpoints) == 4

    servo_min = 1221
    servo_max = 2813
    wheel_radius = 15

    l_2_cmd = -(servo_max - servo_min) / (2 * math.pi / 3 * wheel_radius)

    return [int(setpoints[i] + dls[i] * l_2_cmd) for i in range(4)]


def load_motor_setpoints(filename: str = "../tools/motor_setpoints") -> List[int]:
    file = open(filename, "r")
    values = file.readline()
    return [int(x) for x in values.split(",")]
