Team Members: Raksheta Kulkarni, Norah Wall
Github: RakshetaK, nwall28


## Hardware Setup:
1. Connect Pi Camera
2. Connect ultrasonic sensor triggor pin to GPIO 23 and echo to GPIO 8
3. Connect Servo motor to GPIO 18 (Pin 12)

## Software Setup:
# Install dependencies
pip install torch torchvision opencv-python requests picamera2 RPi.GPIO
# Train face recognition model (one time)
python face_recognition.py

## Running the System:
# Terminal 1: Start web server
python restAPI.py
# Terminal 2: Start cloud tunnel
Create ngrok account and get authtoken.
ngrok config add-authtoken AUTHTOKEN
ngrok http 5000
# Terminal 3: Start detection system
python record.py

# Open browser:
https://haplologic-unsurnamed-kelsi.ngrok-free.dev/

## Libraries used:
- pytorch
- torchvision
- requests
- base64
- RPi.GPIO
- numpy
- faceNetModel