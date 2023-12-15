function output = write_motor_vals(motor_vals, device, DLE, STX, ETX, ACK, ERR)
    % Send packet to arduino containing servo motor positions
    write(device, DLE, "uint8");
    write(device, STX, "uint8");
    write(device, length(motor_vals) * 2, "uint8");
    write(device, motor_vals, "uint16");
    write(device, DLE, "uint8");
    write(device, ETX, "uint8");
    
    % Read from serial port for ACK
    try
        output = read(device, 7, "uint8");
    % Timeout occured, resend the packet
    catch
        disp("timeout");
        write_motor_vals(motor_vals, device, DLE, STX, ETX, ACK, ERR);
    end

    % Error packet was sent (or any other type of packet that isn't ACK)
    if output(5) ~= ACK
        disp("ERR");
        write_motor_vals(motor_vals, device, DLE, STX, ETX, ACK, ERR);
    end
    disp("Motor values sent:");
    disp(motor_vals);
    % Packet was sent and received correctly
end