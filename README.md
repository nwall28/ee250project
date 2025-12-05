Team Members: Raksheta Kulkarni, Norah Wall
Github: RakshetaK, nwall28


## Hardware Setup:
1. Connect Pi Camera
2. Connect ultrasonic sensor
3. Connect Servo motor to GPIO 18 (Pin 12)

## Software Setup:
# Install dependencies
pip install torch torchvision opencv-python requests flask picamera2
# Train face recognition model (one time)
python face_recognition.py

## Running the System:
# Terminal 1: Start web server
python restAPI.py
# Terminal 2: Start detection system
python record.py

# Open browser:
http://localhost:5000/

## Libraries used:
- pytorch
- torchvision
- requests
- base64
- RPi.GPIO
- numpy