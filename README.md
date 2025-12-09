Team Members: Raksheta Kulkarni, Norah Wall
Github: RakshetaK, nwall28


## Hardware Setup:
1. Connect Pi Camera
2. Connect ultrasonic sensor (trigger pin to GPIO 23, and echo to GPIO 8)
3. Connect Servo motor to GPIO 18 (Pin 12)

## Software Setup:
# Install dependencies
python -m venv [your venv name] --system-site-packages
source [your venv name]/bin/activate
pip install torch torchvision opencv-python requests flask picamera2

# if you run into an error that the camera is being used, type the following into your terminal

sudo fuser -v /dev/media*
# kill the process that shows up after this command
sudo kill -9 <PID>

# Use the following to import our kaggle dataset and change the path in train.py
import kagglehub
# Download latest version
path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
print("Path to dataset files:", path)


# Train face recognition model (one time) [we have attached the trained model in facenet_lfw.pt so you don't have to train for over 9 hours]
python3 train.py

# IMPORTANT: if you want to get an embedding vector to make yourself an owner: create a folder called me_images with several normal angles of your face
# Make sure to rename the prototype accordingly and change the name of the PyTorch file loaded in frameExtraction.py

## Running the System:
# Terminal 1: Start web server
python3 restAPI.py
# Terminal 2: Start detection system
python3 record.py

# Open browser:
http://localhost:5000/

## Libraries used:
- pytorch
- torchvision
- requests
- base64
- RPi.GPIO
- numpy