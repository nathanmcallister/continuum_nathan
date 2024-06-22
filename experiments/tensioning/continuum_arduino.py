import numpy as np
import serial
import struct
import math
import time
from typing import List


class ContinuumArduino:
    """
    ContinuumArduino class handles all serial communication with Arduino.
    """

    def __init__(
        self,
        num_motors: int = 4,
        setpoint_filename: str = "../../tools/motor_setpoints",
        wheel_radii: np.ndarray = np.array([15, 15, 15, 15], dtype=float),
        serial_port_name: str = "/dev/ttyACM0",
        timeout: float = 1.0,
    ):
        """
        Creates ContinuumArduino Object and initializes serial port.

        Args:
            num_motors (int): The number of motors in the system
            setpoint_filename (str): Location of motor setpoint text file
            wheel_radii (np.ndarray): Radii of pulley wheels
            serial_port_name (str): The name of the serial port
            timeout: Time in seconds for Serial communication timeout

        Returns:
            ContinuumArduino: A ContinuumArduino object with an active serial port.
        """
        # Motor details
        self.num_motors = num_motors
        self.motor_setpoints = np.array(
            load_motor_setpoints(setpoint_filename), dtype=int
        )
        self.wheel_radii = wheel_radii

        # Servo parameters
        self.servo_min = 1221
        self.servo_max = 2813

        # Transmission flags
        self.transmission_flags = {}
        self.transmission_flags["DLE"] = 0x10
        self.transmission_flags["STX"] = 0x02
        self.transmission_flags["ETX"] = 0x03
        self.transmission_flags["ACK"] = 0x06
        self.transmission_flags["ERR"] = 0x15

        # Packet type flags
        self.packet_flags = {}
        self.packet_flags["CMD"] = 0x04  # Motor command
        self.packet_flags["COM"] = 0x05  # Communication (ACK/ ERR)
        self.packet_flags["NUM"] = 0x07  # Number of motors command
        self.packet_flags["OHZ"] = 0x08  # Oscillator frequency of PWM driver
        self.packet_flags["SHZ"] = 0x09  # Servo (PWM) frequency of PWM driver

        # Transmission error flags
        self.error_flags = {}
        self.error_flags[0x00] = "Length does not match"
        self.error_flags[0x01] = "CRC does not match"
        self.error_flags[0x02] = "Unknown packet flag"

        # Check dimensions
        assert len(self.motor_setpoints) == self.num_motors == len(self.wheel_radii)

        # Initialize Arduino
        self.arduino = serial.Serial(serial_port_name, 115200, timeout=timeout)
        self.__write_num_motors()

    def __build_payload(self, packet_flag: str, data: List[int]) -> bytearray:
        """
        Builds a payload (non transmission/ crc components of packet).

        Args:
            packet_flag: A packet flag denoting what type of packet is to be sent
            data: The data that is to be sent with the packet
        """
        assert flag in self.packet_flags

        payload = bytearray([self.packet_flags[packet_flag]])
        payload.extend(struct.pack("H", 2 * len(data)))

        for datum in data:
            payload.extend(struct.pack("H", datum))

        return payload

    def __build_packet(self, packet_flag: str, data: List[int]) -> bytearray:
        """
        Builds a packet (bytearray) that can then be transmitted.

        Args:
            packet_flag: A packet flag denoting what type of packet is to be sent
            data: The data that is to be sent with the packet

        Returns:
            bytearray: The packet with all transmission bytes, a payload, and a crc, which can be transmitted to the Arduino
        """

        payload = self.build_payload(packet_flag, data)
        crc = self.crc_add_bytes(0, payload)
        packet = bytearray(
            [self.transmission_flags["DLE"], self.transmission_flags["STX"]]
        )

        for byte in payload:
            packet.append(byte)
            if byte == self.transmission_flags["DLE"]:
                packet.append(self.transmission_flags["DLE"])
        packet.extend([self.transmission_flags["DLE"], self.transmission_flags["ETX"]])

        return packet

    def __write_num_motors(self, attempts: int = 5) -> bool:
        """
        Transmit command for Arduino to update the number of motors it controls.

        Args:
            attempts: How many times the system will attempt to send the packet.

        Returns:
            bool: Was the transmission successful?
        """

        packet = self.__build_packet("NUM", [self.num_motors])

        success = False
        attempt = 0
        while not success and attempt < attempts:
            success = self.__transmit_packet(packet)

        return success

    def __transmit_packet(self, packet: bytearray) -> bool:
        """
        Transmit a packet to the arduino, and listen for response.

        Args:
            packet: The packet to be sent (including all transmission bytes and CRC)

        Returns:
            bool: Was the transmission successful?
        """
        # Flush and write packet
        self.arduino.flush()
        self.arduino.write(packet)

        # Read until start sequence is found
        bytes_until_start = self.arduino.read_until(
            bytearray([self.transmission_flags["DLE"], self.transmission_flags["STX"]])
        )

        # No start sequence found
        if not (
            bytes_until_start[-2] == self.transmission_flags["DLE"]
            and bytes_until_start[-1] == self.transmission_flags["STX"]
        ):
            print("Arduino serial timeout: no start sequence found")
            return False

        # Read until end sequence is found
        bytes_until_end = self.arduino.read_until(
            bytearray([self.transmission_flags["DLE"], self.transmission_flags["ETX"]])
        )

        # No end sequence found
        if not (
            bytes_until_end[-2] == self.transmission_flags["DLE"]
            and bytes_until_end[-1] == self.transmission_flags["ETX"]
        ):
            print("Arduino serial timeout: no end sequence found")
            return False

        # Extract payload and crc
        payload = bytes_until_end[:-3]
        crc = bytes_until_end[-3]

        # Expecting a communication type flag
        if payload[0] != self.packet_flags["COM"]:
            print("Non communication packet flag received from Arduino")
            return False

        # Unstuff DLEs
        unstuffed_payload = bytearray()
        dle = False
        for byte in enumerate(payload):
            if dle:
                if byte == self.transmission_flags["DLE"]:
                    unstuffed_payload.append(byte)
                else:
                    unstuffed_payload.extend(
                        bytearray([self.transmission_flags["DLE"], byte])
                    )
                dle = false
            else:
                if byte == self.transmission_flags["DLE"]:
                    dle = True
                else:
                    unstuffed_payload.append(byte)

        # Check packet length match
        expected_length = unstuffed_payload[1]
        data = unstuffed_payload[2:]
        if len(data) != expected_length:
            print(
                "Data received from Arduino does not match the length it specified in the packet"
            )
            return False

        # Check packet length for communication packet specifically
        if expected_length < 2 or expected_length > 3:
            print(
                f"Invalid length ({expected_length}) for ACK/ ERR packet from Arduino"
            )
            return False

        # Compare CRC
        if crc != self.crc_add_bytes(0, unstuffed_payload):
            print("Arduino calculated CRC does not match our CRC (receiving packet).")
            return False

        # At this point, structure of packet is verified
        # Received ACK: successful transmission
        if unstuffed_payload[:2] == bytearray(
            [self.transmission_flags["DLE"], self.transmission_flags["ACK"]]
        ):
            return True
        # Received ERR: unsuccessful transmission
        elif unstuffed_payload[:2] == bytearray(
            [self.transmission_flags["DLE"], self.transmission_flags["ERR"]]
        ):
            print("Arduino sent ERR with the following error code:")
            print(f"{unstuffed_payload[2]}:", self.error_flags[unstuffed_payload[2]])
        # Received unexpected packet
        else:
            print("Arduino sent unexpected packet in response")

        return False

    def write_dls(self, dls: np.ndarray, attempts: int = 5) -> bool:
        """
        Converts cable length deltas (dls) to motor commands, then sends the
        commands over serial to the Arduino, waiting for an acknowledgement.

        Args:
            dls: Changes in cable length from 0 point
            attempts: Number of attempts before deciding transmission is unsuccessful

        Returns:
            bool: Was transmission successful?
        """
        assert len(dls) == self.num_motors

        cmds = (
            (
                self.motor_setpoints
                + (self.servo_max - self.servo_min)
                / (2 * np.pi / 3)
                * dls
                / self.wheel_radii
            )
            .round()
            .astype(int)
        ).tolist()

        packet = self.__build_packet("CMD", cmds)

        # Send packet to Arduino
        success = False
        attempt = 0
        while not success and attempt < attempts:
            success = self.__transmit_packet(packet)
            attempt += 1

        return success

    def update_num_motors(
        self,
        num_motors: int,
        motor_setpoints: np.ndarray,
        wheel_radii: np.ndarray,
        attempts: int = 5,
    ):
        """
        Allows the user to change the number of motors that the system can control.

        Args:
            num_motors: The desired number of motors
            motor_setpoints: The servo setpoints representing 0 cable displacement
            wheel_radii: The radii of the cable pulleys
            attempts: Number of times to try to send the command
        """
        assert num_motors == len(motor_setpoints) == len(wheel_radii)

        self.num_motors = num_motors
        self.motor_setpoints = motor_setpoints
        self.wheel_radii = wheel_radii

        # Send packet to Arduino
        success = False
        attempt = 0
        while not success and attempt < attempts:
            success = self.__write_num_motors()
            attempt += 1

        return success

    def update_oscillator_frequency(self, freq: int, attemps: int = 5) -> bool:
        """
        Update the oscillator frequency of the Adafruit PWM driver.

        Args:
            freq: The desired frequency

        Returns:
            bool: Was the transmission successful?
        """

        packet = self.__build_packet("OHZ", [freq])

        success = False
        attempt = 0
        while not success and attempt < attempts:
            success = self.__transmit_packet(packet)
            attempt += 1

        return success

    def update_servo_frequency(self, freq: int, attemps: int = 5) -> bool:
        """
        Update the servo (PWM) frequency of the Adafruit PWM driver.

        Args:
            freq: The desired frequency

        Returns:
            bool: Was the transmission successful?
        """

        packet = self.__build_packet("SHZ", [freq])

        success = False
        attempt = 0
        while not success and attempt < attempts:
            success = self.__transmit_packet(packet)
            attempt += 1

        return success


"""
Following functions not in class are deprecated, but are included to maintain
compatibility with older scripts.
"""


# Initialize Arduino serial port
def init_arduino():
    return serial.Serial("/dev/ttyACM0", 115200, timeout=1)


# write motor values (commands)
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
            packet.extend(packed_val)
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

    l_2_cmd = (servo_max - servo_min) / (2 * math.pi / 3 * wheel_radius)

    return [int(setpoints[i] + dls[i] * l_2_cmd) for i in range(4)]


def load_motor_setpoints(filename: str = "../tools/motor_setpoints") -> List[int]:
    file = open(filename, "r")
    values = file.readline()
    return [int(x) for x in values.split(",")]
