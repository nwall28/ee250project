import sys, cv2
from pathlib import Path
import numpy as np
from faceNetModel import FaceNet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, models
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from PIL import Image

import base64
import requests
import time

#servo imports
import RPi.GPIO as GPIO

SERVO_PIN = 18 #GPIO pin connected to the servo

def setup_servo():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    pwm = GPIO.PWM(SERVO_PIN, 50) #50Hz frequency
    pwm.start(7.5) #neutral position
    return pwm 


device = torch.device("cpu")

backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
backbone.fc = nn.Identity()

embedding_dim = 128
num_classes = 5749

model = FaceNet(backbone, embedding_dim, num_classes)
model.load_state_dict(torch.load("facenet_lfw.pt",map_location=device))

model.to(device)
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std =[0.5,0.5,0.5]),
    ]
)

def Mp4(dirPath: Path|str, server_url: str = "https://haplologic-unsurnamed-kelsi.ngrok-free.dev"):
    frames1 = process(dirPath) #parse the video and extract the first frame
    #cv2.imwrite("saved_image.jpg",frames1) # use this only if you want to see the image it took of you
    emb = get_Embedded(frames1) #get the vector for the person in the frame
   
    me_bank = torch.load("me_bank.pt", map_location=device)
    emb = get_Embedded(frames1)
    dists = torch.norm(me_bank-emb.unsqueeze(0),p=2,dim=1) #measure the difference between owner embedding vector and person in the frame's embedding vector
    diff = dists.min().item() #get the minimum difference

    THRESH = 1.2
    pwm = setup_servo()

    if(diff<=THRESH):
        print("safe")
        print(diff)
    else:
        print("THIEF!")
        print(diff)
        send_frame_to_server(frames1, server_url) #push image and alert to server
        slap_motion(pwm) #call slap motion function
    
    
def get_Embedded(frame: np.ndarray): #perform necessary transformations on the extracted frame to pass into CNN model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    

    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb, _ = model(x,return_embedding=True)
    return emb.squeeze(0)

def process(path1: Path|str): #extract first frame of video snippet
    frameCount = 0
    capture1 = cv2.VideoCapture(str(path1)) #analyze event

    if not capture1.isOpened():
        raise RuntimeError("Could not open")
    
    # otherwise, processing continues
    print("Processing... (press q to quit)")

    # get video properties
    frameCount = int(capture1.get(cv2.CAP_PROP_FRAME_COUNT))
    if frameCount<=0:
        capture1.release(); 
        raise RuntimeError(f"Empty/invalid video(s): {path1.name}")
   

    # get first frame from video
    capture1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret1, frame1 = capture1.read()
    if not ret1:
        print(f"Failed to read frame {frame1}")
        
    
    capture1.release()
    

    print("\nFrame extraction complete. Ready for ML comparison.")
    return frame1

def send_frame_to_server(frame, server_url):
    try:
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            print("Failed to encode frame")
            return False
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        payload = {
            'image': img_base64,
            'format': 'jpeg'
            }
        print(f"Sending alert to {server_url}...")
        response = requests.post(server_url, json=payload, timeout = 10)

        if response.status_code == 201:
            print("Alert sent successfully.")
            return True
        else:
            print(f"Failed to send alert. Status code: {response.status_code}")
            return False
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server.")
        return False
    except Exception as e:
        print(f"Error sending frame to server: {e}")
        return False

def slap_motion(pwm):
    print("Slap!")
    pwm.ChangeDutyCycle(2.5) #left
    time.sleep(0.2)
    pwm.ChangeDutyCycle(12.5) #right
    time.sleep(0.2)
    pwm.ChangeDutyCycle(7.5) #neutral
    time.sleep(0.2)
    pwm.ChangeDutyCycle(0) #stop
 

def cleanup_servo(pwm):
    """Clean up GPIO"""
    if pwm is not None:
        pwm.stop()
        GPIO.cleanup()
        print("GPIO cleaned up")
            
        

