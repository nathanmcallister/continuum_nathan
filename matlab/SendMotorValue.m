%% SendMotorValues.m - Sends a packet of information (handling all DLEs and flags) to the arduino
% Ensures that packet was received properly through the use of an ack
% 
% Cam Wolfe 11/22/2023
clear all;

SERIALPORT_ARDUINO = "COM3";

SERVO_MIN = 80;
SERVO_MAX = 530;
TIGHTENING_FACTOR = -30;
SERVO_MID = (SERVO_MIN + SERVO_MAX)/2;


% setpoints = [SERVO_MIN:50:SERVO_MAX]';
% setpoints = [SERVO_MIN]';
% setpoints = [SERVO_MAX]';
% setpoints = SERVO_MID;
setpoints = (SERVO_MID) * ones(1,4);
%setpoints = [350 365 430 390];
%setpoints = [100 100 100 100; 500 500 500 500];

m1_offset = -10;
m2_offset = 35;
m3_offset = 40;
m4_offset = 20;

setpoint = SERVO_MID + [m1_offset, m2_offset, m3_offset, m4_offset];

motor_values = setpoint + [1 0 -1 0] * 00;

m4_setpoint = 390;

% Serial packets
DLE = 0x10;
STX = 0x12;
ETX = 0x13;
ACK = 0x14;
ERR = 0x15;

%Establish serial connection
device = serialport(SERIALPORT_ARDUINO, 115200);
flush(device);
for j=1:1
    for i=1:size(setpoints,1)
        write_motor_vals(motor_values(i,:), device, DLE, STX, ETX, ACK, ERR);
        pause(1);
    end
end
clear device;