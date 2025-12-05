import sys, cv2
from pathlib import Path
from collections import Counter
import numpy as np
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



def Mp4(dirPath: Path, server_url = "http://localhost:5000/image"):
    pwm = setup_servo()
    frames1 = process(dirPath)
    name = faceRec(frames1)
    if(name!="Raksheta"):
        print("thief!")
        #servo slap stick code here
        #have it push something to the website
        send_frame_to_server(frames1, server_url) #push image and alert to server
        slap_motion(pwm) #call slap motion function
    else:
        print("safe")
    cleanup_servo(pwm)
    
    

def process(path1: Path):
    frameCount = 0
    capture1 = cv2.VideoCapture(str(path1)) #analyze key event 1

    if not capture1.isOpened():
        raise RuntimeError("Could not open")
    
    # otherwise, processing continues
    print("Processing... (press q to quit)")

    # get video properties
    frameCount = int(capture1.get(cv2.CAP_PROP_FRAME_COUNT))
    if frameCount<=0:
        capture1.release(); 
        raise RuntimeError(f"Empty/invalid video(s): {path1.name}")
   

    # get first frame, middle frame, and last frame for both clips (0, 255, 450th frame)
    
    
    capture1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret1, frame1 = capture1.read()
    if not ret1:
        print(f"Failed to read frame pair {frame1}")
        
    
    capture1.release()
    
    # cv2.destroyAllWindows()

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
            
        
# def main():
    # l1,l2 = process("Video_1 (1).mov","Video (1).mov")
    # removed_list = frame_difference(l1,l2)
    # print("Items removed", removed_list)
# if __name__ == "__main__":
#     main()

