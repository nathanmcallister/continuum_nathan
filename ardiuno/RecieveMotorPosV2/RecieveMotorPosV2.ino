#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Serial definitions
#define DLE 0x61
#define STX 0x62
#define ETX 0x63
#define ACK 0x64
#define ERR 0x65

// Servo definitions
#define NUM_SERVOS 4
#define SERVOMIN 100   // This is the 'minimum' pulse length count (out of 4096)
#define SERVOMAX 500   // This is the 'maximum' pulse length count (out of 4096)
#define SWEEPLEN 125   // This is the max distance from middle the servo will go
#define SERVOMID 300   // This is the 'middle' pulse length count (out of 4096)
#define USMIN 600      // This is the rounded 'minimum' microsecond length based on the minimum pulse of 150
#define USMAX 2400     // This is the rounded 'maximum' microsecond length based on the maximum pulse of 600
#define SERVO_FREQ 50  // Analog servos run at ~50 Hz updates

// PWM driver
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// State variables
bool dle_high = false;
bool reading = false;
int counter = 0;
uint8_t buf[64];

// Data packet
typedef struct {
    uint8_t data_length;
    uint8_t* data_pointer;
} data_t;

void setup() {
    // Start serial
    SerialUSB.begin(115200);
    
    // Start PWM driver
    pwm.begin();
    pwm.setOscillatorFrequency(27000000);
    pwm.setPWMFreq(SERVO_FREQ);

    delay(10);
}

void loop() {
    while (SerialUSB.available()) {

        // Read from serial
        uint8_t value = SerialUSB.read();
        
        // Put into buffer
        buf[counter] = value;
        counter++;

        // Reading mode is after a start series (DLE STX) and before an end series (DLE ETX)
        if (reading) {

            // Last byte recieved was a DLE byte
            if (dle_high) {

                // End transmission
                if (value == ETX) {

                    // Exit reading mode
                    reading = false;

                    // Parse packet
                    data_t* data_packet = parse_serial_data(counter);
                    
                    // Packet was recieved properly
                    if (data_packet != NULL) {

                        // Send acknowledgement
                        send_ack();

                        // Motor packet has data_length of 2 * # of servos
                        if (data_packet->data_length == 2 * NUM_SERVOS) {
                            move_motors(data_packet);
                        }

                        // Handle malloc
                        free_data_pointer(data_packet);
                    }

                    // Error in data packet
                    else {
                        send_error();
                    }
                }

                // Lower DLE flag
                dle_high = false;
            }

            if (!dle_high) {

                // Handle DLE flags
                if (value == DLE) {
                    dle_high = true;
                }
            }
        }

        // Not in reading mode
        else {

            // Raise dle_flag
            if (value == DLE) {
                dle_high = true;
            }

            else {

                // Start sequence received
                if (dle_high && value == STX) {

                    // Start writing into beginning of buffer
                    buf[0] = DLE;
                    buf[1] = STX;
                    counter = 2;
                    reading = true;
                }

                dle_high = false;
            }
        }
    }
}

// Handle freeing heap memory for data structures
void free_data_pointer(data_t* data) {

    // Don't free NULL pointers
    if (data == NULL) {
        SerialUSB.println("Tried to free a data packet that was NULL.");
    }

    // Free heap memory
    else {
        free(data->data_pointer);
        free(data);
    }
}

// Take raw byte string and parse into data packet
data_t* parse_serial_data(int end_of_packet) {

    // Data struct pointer
    data_t* data = NULL;

    // State variables
    bool dle = false;
    bool data_reading = false;
    int packet_length = -1;
    int data_counter = 0;

    // Go through byte string
    for (int i = 0; i < end_of_packet; i++) {
        uint8_t value = buf[i];

        // Not hit start byte yet
        if (!data_reading) {

            // DLE flag is high
            if (dle) {

                // Start byte
                if (value == STX) {
                    data_reading = true;
                }

                // Lower DLE flag
                dle = false;
            }

            // DLE flag is low
            else {

                // Raise DLE flag
                if (value == DLE) {
                    dle = true;
                }
            }
        }
        
        // Reading data
        else {

            // Packet length hasn't been read yet (directly after start byte)
            if (packet_length == -1) {

                // DLE flag is high
                if (dle) {

                    // Double DLE flag means packet length has the same value as DLE byte
                    if (value == DLE) {
                        packet_length = value;
                        dle = false;
                    }

                    // Should not receive a flag for length, meaning that DLE is not good
                    else {
                        return NULL;
                    }
                }

                // DLE flag is low
                else {

                    // Raise DLE flag
                    if (value == DLE) {
                        dle = true;
                    }

                    // Byte after start byte is packet length, so apply it
                    else {
                        packet_length = value;
                        data = (data_t*) malloc(sizeof(data_t));
                        data->data_length = (uint8_t) packet_length;
                        data->data_pointer = (uint8_t*) calloc(packet_length, sizeof(uint8_t));
                    }
                }
            }

            // Packet length is set
            else {

                // DLE flag is high
                if (dle) {

                    // End of transmission
                    if (value == ETX) {

                        // Must have lost some bytes or there was an error in the packet length, return NULL pointer
                        if (data_counter != packet_length) {
                            free_data_pointer(data);
                            return NULL;
                        }

                        // Valid data packet, return pointer to struct
                        else {
                            return data;
                        }
                    }

                    // Overrun packet length, this is an error
                    if (data_counter >= packet_length) {
                        free_data_pointer(data);
                        return NULL;
                    }

                    // Store value in array
                    data->data_pointer[data_counter] = value;
                    data_counter ++;
                    dle = false;
                }

                // DLE flag is low
                else {

                    // Raise DLE flag
                    if (value == DLE) {
                        dle = true;
                    }

                    // Non-DLE (data) byte
                    else {

                        // Overrun packet length, this is an error
                        if (data_counter >= packet_length) {
                            free_data_pointer(data);
                            return NULL;
                        }

                        // Store value in array
                        data->data_pointer[data_counter] = value;
                        data_counter ++;
                    }
                }
            }
        }
    }

    // No end of packet found, this is an error
    free_data_pointer(data);
    return NULL;
}

void move_motors(data_t* data) {

    // Go through data packet
    for (int i = 0; i < NUM_SERVOS; i++) {

        // Convert 2 little-endian uint8_t's to one uint16_t
        uint16_t motor_command = ((uint16_t) data->data_pointer[2*i]) | ((uint16_t) data->data_pointer[2*i+1] << 8);
        
        // Send to pwm chip
        pwm.setPWM(i, 0, motor_commands[i]);
    }
}

// Acknowledgement that packet was received correctly
void send_ack() {
    SerialUSB.write(DLE);
    SerialUSB.write(STX);
    SerialUSB.write(1);
    SerialUSB.write(DLE);
    SerialUSB.write(ACK);
    SerialUSB.write(DLE);
    SerialUSB.write(ETX);
}

// Error in packet transmission
void send_error() {
    SerialUSB.write(DLE);
    SerialUSB.write(STX);
    SerialUSB.write(1);
    SerialUSB.write(DLE);
    SerialUSB.write(ERR);
    SerialUSB.write(DLE);
    SerialUSB.write(ETX);
}