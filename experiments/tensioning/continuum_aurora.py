import numpy as np
import time
import serial
import struct
import kinematics
from typing import List, Dict


def init_aurora():
    return serial.Serial(
        "/dev/ttyUSB0",
        115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1,
        xonxoff=False,
    )


def crc_add_bytes(CRC: int, byte_array):
    CRC = CRC & 0xFF  # Ensure CRC is treated as uint8
    for byte in byte_array:
        for bit_num in range(8, 0, -1):
            thisBit = (byte >> (bit_num - 1)) & 1
            doInvert = (thisBit ^ ((CRC & 128) >> 7)) == 1
            CRC = (CRC << 1) & 0xFF  # Ensure the result is treated as uint8
            if doInvert:
                CRC = CRC ^ 7
    return CRC


def request_aurora_packet(serial_port: serial.Serial, probe_sn_array: List[str]):
    # Definitions
    DLE_BYTE = 0x10
    STX_BYTE = 0x02
    ETX_BYTE = 0x03
    GET_PROBE_TFORM = 0x13

    # Assemble payload
    payload = bytearray()
    payload.append(GET_PROBE_TFORM)  # packet type
    payload.append(len(probe_sn_array))  # sent as uint8_t

    # Add port handle string characters
    for probe_sn in probe_sn_array:
        payload_insert = bytearray([0x00, 0x00, 0x00, 0x00])
        this_porthandle_string = bytearray(probe_sn, "utf-8")
        assert len(this_porthandle_string) == 2, "Port handle string length must be 2!"
        payload_insert[: len(this_porthandle_string)] = this_porthandle_string
        payload.extend(payload_insert)

    # Compute CRC
    CRC_BYTE = crc_add_bytes(0, payload)
    payload.append(CRC_BYTE)  # CRC byte IS stuffed... (TODO: check this)

    # Assemble message with DLE stuffing
    msg = bytearray([DLE_BYTE, STX_BYTE])  # first two bytes not stuffed
    for byte in payload:
        msg.append(byte)
        if byte == DLE_BYTE:  # handle DLE stuffing
            msg.append(DLE_BYTE)
    msg.extend([DLE_BYTE, ETX_BYTE])

    # Write message
    serial_port.write(msg)


def get_aurora_packet(serial_port, timeout):
    # Initial setup
    DLE_BYTE = 0x10
    STX_BYTE = 0x02
    ETX_BYTE = 0x03
    PKT_TYPE_TRANSFORM_DATA = 0x01
    message_max_size = 300
    new_serial_data = bytearray()
    start_idx = None

    # Record the start time
    start_time = time.time()

    # Flush serial port
    serial_port.reset_input_buffer()

    # Keep reading data until we find the start pattern or timeout
    while start_idx is None and (time.time() - start_time) < timeout:
        if serial_port.in_waiting > 0:
            data = serial_port.read(serial_port.in_waiting)
            new_serial_data.extend(data)

            # Search for the start of the message
            start_idx = new_serial_data.find(bytearray([DLE_BYTE, STX_BYTE]))

            # If found, trim the data
            if start_idx != -1:
                new_serial_data = new_serial_data[start_idx:]

    # Exit if start pattern not found within timeout
    if start_idx is None:
        print("Timeout or start pattern not found.")
        return None

    # Now, we are looking for the end pattern within the timeout
    pkt = None
    while pkt is None and (time.time() - start_time) < timeout:
        end_idx = new_serial_data.find(bytearray([DLE_BYTE, ETX_BYTE]))
        if end_idx != -1:
            # Extract the message, considering DLE stuffing
            message = bytearray()
            i = 0
            while i < end_idx + 2:  # Include ETX_BYTE
                message.append(new_serial_data[i])
                # Skip the stuffed DLE byte
                if (
                    new_serial_data[i] == DLE_BYTE
                    and i + 1 < len(new_serial_data)
                    and new_serial_data[i + 1] == DLE_BYTE
                ):
                    i += 1
                i += 1

            # Here you would parse `message` as per your protocol

            # Remove the processed message from buffer
            new_serial_data = new_serial_data[end_idx + 2 :]
            pkt = message  # Placeholder, replace with actual parsing result

        # Read more data if needed
        if serial_port.in_waiting > 0:
            data = serial_port.read(serial_port.in_waiting)
            new_serial_data.extend(data)

    if (time.time() - start_time) >= timeout:
        print("Serial read timeout!")

    return pkt


def unstuff_dle(pkt: bytearray) -> bytearray:
    DLE = 0x10
    output = b""

    dle_flag = False
    for i in range(len(pkt)):
        if pkt[i] == DLE:
            if not dle_flag:
                dle_flag = True
            else:
                dle_flag = False
                output += int.to_bytes(pkt[i], 1, "little")
        else:
            if dle_flag:
                output += int.to_bytes(pkt[i - 1], 1, "little")
                dle_flag = False
            output += int.to_bytes(pkt[i], 1, "little")

    return output


def parse_aurora_transforms(pkt: bytearray) -> Dict[str, List[float]]:

    # Get rid of stuffed DLE's
    pkt = unstuff_dle(pkt)

    # Ensure we got a transform (can add error handling later)
    assert pkt[2] == 0x01

    # Parse data packet
    num_transforms = pkt[3]
    frame = int.from_bytes(pkt[4:8], "little")
    transform_dict = {}

    # Go through all tools and get transform data
    for i in range(num_transforms):
        data_bytes = pkt[8 + 36 * i : 8 + 36 * (i + 1) + 1]
        tool_id = data_bytes[0:2].decode("utf-8")
        transform_data = [0.0] * 8
        for j in range(8):
            transform_data_bytes = data_bytes[4 + 4 * j : 4 + 4 * (j + 1)]
            transform_data[j] = struct.unpack("f", transform_data_bytes)[0]

        transform_dict[tool_id] = transform_data

    return transform_dict


def get_aurora_transforms(
    serial_port: serial.Serial, probe_list: List[str], timeout: float = 1
) -> Dict[str, List[float]]:
    output = {}
    while not output:
        request_aurora_packet(serial_port, probe_list)
        pkt = get_aurora_packet(serial_port, timeout)
        output = parse_aurora_transforms(pkt)
    return output


def aurora_transform_2_T(aurora_transform: List[float]) -> np.ndarray:
    T = np.identity(4)

    q = np.array(aurora_transform[0:4])
    t = np.array(aurora_transform[4:-1])

    R = kinematics.quat_2_dcm(q)

    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T


def get_T_tip_2_model(
    aurora_transform: List[float],
    T_aurora_2_model: np.ndarray,
    T_tip_2_coil: np.ndarray,
) -> np.ndarray:
    T_coil_2_aurora = aurora_transform_2_T(aurora_transform)

    return np.matmul(T_aurora_2_model, np.matmul(T_coil_2_aurora, T_tip_2_coil))


