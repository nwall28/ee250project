import sys, cv2
from pathlib import Path
from collections import Counter
import numpy as np



def Mp4(dirPath: Path):
    frames1 = process(dirPath)
    name = faceRec(frames1)
    if(name!="Raksheta"):
        print("thief!")
        #servo slap stick code here
        #have it push something to the website
    else:
        print("safe")
    
    

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


            
        
# def main():
    # l1,l2 = process("Video_1 (1).mov","Video (1).mov")
    # removed_list = frame_difference(l1,l2)
    # print("Items removed", removed_list)
# if __name__ == "__main__":
#     main()



























