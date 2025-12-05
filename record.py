from pathlib import Path
from collections import deque
import tempfile, time, datetime as dt


from frameExtraction import Mp4 # import the process function

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput #mp4 cotainer needs ffmpeg

RESOLUTION = (1920,1080)
FPS = 30
RECORD_SECONDS = 4 #fix this later because our system isn't a set number of seconds
CLIP_DIR = Path(tempfile.gettempdir())/"pi_clips"
CLIP_DIR.mkdir(parents=True, exists = True)

class UltrasonicTrigger:
    '''This is just a placeholder right now for when we figure out
    the details for the ultrasonic GPIO pins and what stalls the video from being recorded'''
    def trigger_function(self):
        return
    
'''creates a unique timestamp based filename for each recorded video'''
def ts_name(prefix = "clip", ext = ".mp4"):
    return f"{prefix}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"

'''declares and configures the camera to a fixed resolution, format'''
def build_cam():
    cam = Picamera2()
    cfg = cam.create_video_configuration(
        main={"size": RESOLUTION, "format": "RGB888"},
        controls={"FrameDurationLimits": (int(1e6/FPS), int(1e6/FPS))}
    )
    cam.configure(cfg)
    return cam

'''starts the camera recording to the exact file path that we passed in'''
def record_mp4(cam: Picamera2, output_path: Path, seconds: int):
    enc = H264Encoder(bitrate=8_000_000)
    out = FfmpegOutput(str(output_path))
    cam.start_recording(encoder = enc, output=out)
    try:
        time.sleep(seconds)
    finally:
        cam.stop_recording()

'''Creates a unique, tampestamped filename in the clip directory through ts_name(), then calls
record_mp4 with that path and the seconds of the video (FIX THIS), returning the path to this saved clip'''
def capture_event(cam:Picamera2) -> Path:
    p = CLIP_DIR / ts_name()
    record_mp4(cam,p,RECORD_SECONDS) #instance of frame length here -> FIX
    return p

'''the main function declares an ultrasonic instance as the trigger for the recording events,
the camera variable, tells us where the clip will be saved, creates a "memory" buffer and runs the loop to wait for trigger, 
start the camera, call the process function, and ends the process'''
def main():
    # ultrasonic = UltrasonicTrigger()
    cam = build_cam()
    cam.start()

    print(f"saving file to {CLIP_DIR}")
    

    try:
        # while True:
            # ultrasonic.trigger_function()
        clip = capture_event(cam)
        print(f"saved: {clip}")
        Mp4(clip)
            
    except KeyboardInterrupt:
        print("Exiting from issues")
    finally:
        cam.stop()
        print("Ended camera process")
if __name__ == "__main__":
    main()