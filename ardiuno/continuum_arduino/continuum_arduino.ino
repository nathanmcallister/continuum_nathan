#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Transmission flags
#define DLE 0x10
#define STX 0x02
#define ETX 0x03
#define ACK 0x06
#define ERR 0x15

// Packet type flags
#define CMD 0x04 // Motor command
#define COM 0x05 // Communication (ACK/ ERR)
#define NUM 0x07 // Number of motors command
#define OHZ 0x08 // Oscillator frequency of PWM driver
#define SHZ 0x09 // Servo (PWM) frequency of PWM driver

// Error codes
#define LENGTH_ERROR 0x01 // Length does not match
#define CRC_ERROR 0x02 // CRC does not match
#define TYPE_ERROR 0x03 // Unknown or incorrect packet flag
#define UNINITIALIZED_ERROR 0x04 // System has not been initialized (number of servos and frequencies set)
#define NUM_MOTOR_ERROR 0x05 // Number of motors does not match command sent
#define PARAM_LENGTH_ERROR 0x06 // Parameter update packet length is incorrect
#define PARAM_VALUE_ERROR 0x07 // Invalid parameter value sent
#define COM_ERROR 0x08 // Invalid communication packet

// Servo definitions
#define SERVOMIN 1221   // This is the 'minimum' pulse length count (out of 4096)
#define SERVOMAX 2813   // This is the 'maximum' pulse length count (out of 4096)

// Testing
#define TESTING false

void send_ack();
void send_error(uint8_t error_code);
uint8_t crc_add_bytes(uint8_t crc, uint8_t* payload, int length);
int unstuff_dle(int length);

// System properties
int num_servos = 0;
int oscillator_frequency = 0;
int servo_frequency = 0;

// PWM driver
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// State variables
bool dle_high = false;
bool reading = false;
int counter = 0;
uint8_t buf[64];

void setup() {
    // Start serial
    Serial.begin(115200);
    
    // Start PWM driver
    if (!TESTING) {
      pwm.begin();
    } 
    /*
    * In theory the internal oscillator (clock) is 25MHz but it really isn't
    * that precise. You can 'calibrate' this by tweaking this number until
    * you get the PWM update frequency you're expecting!
    * The int.osc. for the PCA9685 chip is a range between about 23-27MHz and
    * is used for calculating things like writeMicroseconds()
    * Analog servos run at ~50 Hz updates, It is importaint to use an
    * oscilloscope in setting the int.osc frequency for the I2C PCA9685 chip.
    * 1) Attach the oscilloscope to one of the PWM signal pins and ground on
    *    the I2C PCA9685 chip you are setting the value for.
    * 2) Adjust setOscillatorFrequency() until the PWM update frequency is the
    *    expected value (50Hz for most ESCs)
    * Setting the value here is specific to each individual I2C PCA9685 chip and
    * affects the calculations for the PWM update frequency. 
    * Failure to correctly set the int.osc value will cause unexpected PWM results
    */
}

void loop() {
    if (Serial.available()) {
        uint8_t value = Serial.read();
        if (!reading) {
            if (!dle_high) {
                if (value == DLE) {
                    dle_high = true;
                }
            }
            else {
                if (value == STX) {
                    reading = true;
                    counter = 0;
                }
                dle_high = false;
            }
        }
        else {
            buf[counter] = value;
            if (!dle_high) {
                if (value == DLE) {
                    dle_high = true;
                }
            }
            else {
                if (value == ETX) {
                    int payload_length = counter - 1; // length = counter + 1, payload_length = length - 2
                    parse_payload(payload_length);

                    reading = false;
                    counter = 0;
                }
                dle_high = false;
            }
            if (reading) {
                counter++;
            }
        }
    }
}

void parse_payload(int length) {
    length = unstuff_dle(length);

    uint8_t flag = buf[0];
    uint8_t data_length = buf[1];

    // Assert data length is equal to number of data bytes (total number of bytes minus the flag and data_length, and crc bytes)
    if (length - 3 != data_length) {
        send_error(LENGTH_ERROR);
        return;
    }

    uint8_t packet_crc = buf[length-1];
    uint8_t crc = crc_add_bytes(0, buf, length - 1);
    if (crc != packet_crc) {
        send_error(crc);
        return;
    }

    // Motor command
    if (flag == CMD) {
        // See if system is initialized before updating motors
        bool num_servos_uninitialized = num_servos == 0;
        bool oscillator_frequency_uninitialized = oscillator_frequency == 0;
        bool servo_frequency_uninitialized = servo_frequency == 0;
        if (num_servos_uninitialized || oscillator_frequency_uninitialized || servo_frequency_uninitialized) {
            send_error(UNINITIALIZED_ERROR);
            return;
        }

        if (2 * num_servos != data_length) {
            send_error(NUM_MOTOR_ERROR);
            return;
        }
        // System is initialized, write to motors
        for (int i = 0; i < num_servos; i++) {
            uint16_t motor_command = ((uint16_t) buf[2*(i+1)]) | (((uint16_t) buf[2*(i+1)+1]) << 8);
            if (!TESTING) {
              pwm.setPWM(i, 0, min(SERVOMAX, max(SERVOMIN, motor_command)));
            }
        }

        send_ack();
    }
    // Communication packet, just echo what was sent (assumes proper structure {COM, 0x02, DLE, ACK} or {COM 0x03, DLE, ERR, CODE})
    else if (flag == COM) {
        if (buf[3] == ACK) {
            send_ack();
        }
        else if (buf[3] == ERR) {
            send_error(buf[4]);
        }
        else {
            send_error(COM_ERROR);
        }
    }
    // Update the number of servos to control
    else if (flag == NUM) {
        if (data_length != 2) {
            send_error(PARAM_LENGTH_ERROR);
            return;
        }

        uint16_t new_num_servos = ((uint16_t) buf[2]) | (((uint16_t) buf[3]) << 8);

        if (new_num_servos == 0 || new_num_servos > 16) {
            send_error(PARAM_VALUE_ERROR);
            return;
        }
        num_servos = new_num_servos;
        send_ack();
    }
    // Update the oscillator clock frequency
    else if (flag == OHZ) {
        if (data_length != 2) {
            send_error(PARAM_LENGTH_ERROR);
            return;
        }

        uint16_t new_oscillator_frequency = (((uint16_t) buf[2]) | (((uint16_t) buf[3]) << 8)); // Unpack and convert value in kHz to Hz

        if (new_oscillator_frequency < 23000 || new_oscillator_frequency > 27000) {
            send_error(PARAM_VALUE_ERROR);
            return;
        }
        oscillator_frequency = new_oscillator_frequency * 1000;
        if (!TESTING) {
          pwm.setOscillatorFrequency(oscillator_frequency);
        }
        send_ack();
    }
    // Update the servo (PWM) frequency
    else if (flag == SHZ) {
        if (data_length != 2) {
            send_error(PARAM_LENGTH_ERROR);
            return;
        }

        uint16_t new_servo_frequency = (((uint16_t) buf[2]) | (((uint16_t) buf[3]) << 8)); // Unpack and convert value in kHz to Hz

        if (new_servo_frequency < 40 || new_servo_frequency > 1600) {
            send_error(PARAM_VALUE_ERROR);
            return;
        }
        servo_frequency = new_servo_frequency;
        if (!TESTING) {
          pwm.setPWMFreq(servo_frequency);
        }

        send_ack();
    }
    // Unknown packet type
    else {
        send_error(TYPE_ERROR);
    }
}

// Acknowledgement that packet was received correctly
void send_ack() {
    uint8_t payload[4] = {COM, 0x02, DLE, ACK};
    uint8_t crc = crc_add_bytes(0, payload, 4);
    Serial.write(DLE);
    Serial.write(STX);
    for (int i = 0; i < 4; i++) {
        Serial.write(payload[i]);
    }
    Serial.write(crc);
    Serial.write(DLE);
    Serial.write(ETX);
}

// Error in packet transmission
void send_error(uint8_t error_code) {
    uint8_t payload[5] = {COM, 0x03, DLE, ERR, error_code};
    uint8_t crc = crc_add_bytes(0, payload, 5);
    Serial.write(DLE);
    Serial.write(STX);
    for (int i = 0; i < 5; i++) {
        Serial.write(payload[i]);
    }
    Serial.write(crc);
    Serial.write(DLE);
    Serial.write(ETX);
}

uint8_t crc_add_bytes(uint8_t CRC, uint8_t* payload, int length) {
    CRC = CRC & 0xFF;  // Ensure CRC is treated as uint8_t
    for (int i = 0; i < length; i++) {
        uint8_t byte = payload[i];
        for (int bit_num = 8; bit_num > 0; bit_num--) {
            uint8_t thisBit = (byte >> (bit_num - 1)) & 1;
            uint8_t doInvert = (thisBit ^ ((CRC & 128) >> 7)) == 1;
            CRC = CRC << 1;
            if (doInvert) {
                CRC = CRC ^ 7;
            }
        }
    }
    return CRC;
}

int unstuff_dle(int length) {
    int counter = 0;
    bool dle = false;
    for (int i = 0; i < length; i++) {
        if (dle) {
            buf[counter] = DLE;
            counter++;
            if (buf[i] != DLE) {
                buf[counter] = buf[i];
                counter++;
            }
            dle = false;
        }
        else {
            if (buf[i] == DLE) {
                dle = true;
            }
            else {
                buf[counter] = buf[i];
                counter++;
            }
        }
    }

    return counter;
}
