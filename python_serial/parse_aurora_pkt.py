#!/bin/python3
import struct
from typing import Dict, List

def get_aurora_transforms(pkt: bytearray) -> Dict[str, List[float]]:

    # We got a transform
    if pkt[2] == 1:

        num_transforms = pkt[3]
        frame = int.from_bytes(pkt[4:8], 'little')
        transform_dict = {}
        for i in range(num_transforms):
            data_bytes = pkt[8+36*i:8+36*(i+1)+1]
            tool_id = data_bytes[0:2].decode("utf-8")
            transform_data = [0.0]*8
            for j in range(8):
                transform_data_bytes = data_bytes[4+4*j:4+4*(j+1)]
                transform_data[j] = struct.unpack('f', transform_data_bytes)[0]

            transform_dict[tool_id] = transform_data

        return transform_dict


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
                output += pkt[i]
        else:
            if dle_flag:
                dle_flag = False
            output += pkt[i]


test = b"\x10\x02\x01\x10\x10\x10\x03"
print(unstuff_dle(test))






