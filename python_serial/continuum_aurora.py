import time
import serial
from typing import List


def crc_add_bytes(CRC, byte_array):
    CRC = CRC & 0xFF  # Ensure CRC is treated as uint8
    for byte in byte_array:
        for bit_num in range(8, 0, -1):
            thisBit = (byte >> (bit_num - 1)) & 1 
            doInvert = (thisBit ^ ((CRC & 128) >> 7)) == 1
            CRC = (CRC << 1) & 0xFF  # Ensure the result is treated as uint8
            if doInvert:
                CRC = CRC ^ 7
    return CRC


def request_aurora_packet(
    serial_port: serial.Serial, probe_sn_array: List[str]
):
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
        this_porthandle_string = bytearray(probe_sn, 'utf-8')
        assert (
            len(this_porthandle_string) == 2
        ), "Port handle string length must be 2!"
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
            new_serial_data = new_serial_data[end_idx + 2:]
            pkt = message  # Placeholder, replace with actual parsing result

        # Read more data if needed
        if serial_port.in_waiting > 0:
            data = serial_port.read(serial_port.in_waiting)
            new_serial_data.extend(data)

    if (time.time() - start_time) >= timeout:
        print("Serial read timeout!")

    return pkt
