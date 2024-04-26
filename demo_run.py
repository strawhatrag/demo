import serial
import subprocess
import time

# Define the serial ports and baud rates for the two Arduino boards
serial_port_1 = '/dev/ttyUSB0'  # Adjust as needed for the first Arduino
serial_port_2 = '/dev/ttyUSB1'  # Adjust as needed for the second Arduino
baud_rate = 9600

# Function to upload .ino file to Arduino
def upload_ino_file(file_path, serial_port):
    # Compile the .ino file
    compile_cmd = f"arduino-cli compile --fqbn arduino:avr:uno {file_path}"
    compile_result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
    if compile_result.returncode != 0:
        print(f"Failed to compile {file_path}: {compile_result.stderr}")
        return False

    # Upload the compiled sketch
    upload_cmd = f"arduino-cli upload -p {serial_port} --fqbn arduino:avr:uno {file_path}"
    upload_result = subprocess.run(upload_cmd, shell=True, capture_output=True, text=True)
    if upload_result.returncode != 0:
        print(f"Failed to upload {file_path} to {serial_port}: {upload_result.stderr}")
        return False

    print(f"Successfully uploaded {file_path} to {serial_port}")
    return True

# Function to check if Arduino has finished its task
def arduino_finished(arduino):
    # You can customize this function based on how the Arduino indicates it has finished
    # For example, waiting for a specific response or a timeout
    timeout = 60  # Time in seconds
    start_time = time.time()
    while time.time() - start_time < timeout:
        if arduino.in_waiting > 0:
            line = arduino.readline().decode().strip()
            print(f"Arduino 1: {line}")
            if "finished" in line:  # Adjust this condition as needed
                return True
    return False

# Create serial connections to the two Arduino boards
arduino_1 = serial.Serial(serial_port_1, baud_rate)
arduino_2 = serial.Serial(serial_port_2, baud_rate)
time.sleep(2)  # Allow time for the connections to be established

# Upload and execute 'Line_follower_app.ino' on Arduino 1
if upload_ino_file('/path/to/Line_follower_app.ino', serial_port_1):
    print(f"Executing 'Line_follower_app.ino' on Arduino connected to {serial_port_1}")
    # Wait for Arduino 1 to finish its execution
    if arduino_finished(arduino_1):
        print("Arduino 1 has finished its execution")

# Upload and execute 'inverse_2.ino' on Arduino 2
if upload_ino_file('/path/to/inverse_2.ino', serial_port_2):
    print(f"Executing 'inverse_2.ino' on Arduino connected to {serial_port_2}")
    # Optionally, you can read and print data from Arduino 2
    while arduino_2.in_waiting > 0:
        print(arduino_2.readline().decode().strip())

# Close the serial connections
arduino_1.close()
arduino_2.close()
