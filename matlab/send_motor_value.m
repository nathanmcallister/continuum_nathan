%% send_motor_values.m - Sends a packet of information (handling all DLEs and flags) to the arduino
% Ensures that packet was received properly through the use of an ack
% 
% Cam Wolfe 11/22/2023
clear all;

SERIALPORT_ARDUINO = "COM3";

setpoints = [100; 200; 300; 400; 500];

motor_values = setpoints * ones(1,4);

% Serial packets
DLE = 0x10;
STX = 0x12;
ETX = 0x13;
ACK = 0x14;
ERR = 0x15;

%Establish serial connection
device = serialport(SERIALPORT_ARDUINO, 115200);
flush(device);

for i=1:length(setpoints)
    write_motor_vals(motor_values(i,:), device, DLE, STX, ETX, ACK, ERR);
    pause(1);
end
clear device;
