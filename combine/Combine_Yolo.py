import numpy as np
import imutils
import time as t
import json
import cv2
from demog import*
from no_plate_detection_video import*
from yolo_edit import*
from yolo_video import*
from Fire_detection_video import*
from obj_video import*
cap = cv2.VideoCapture("18.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('frames per second : ' + str(fps))
print('Total number of frames : ' + str(frame_count))
output={}

def _main():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Done")
            break
        time = cap.get(cv2.CAP_PROP_POS_MSEC)
        time = round(time / 1000, 2)
        frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
        finaldict = {}
        #finaldict["frame_no"] = frame_no
        finaldict["Time"]= time
        finaldict["persons dict"] = persons(frame)
        finaldict["Demographics"]= demographics(frame)
        finaldict["Dominant color"]=color_detection(frame)
        finaldict["Animals"]=animal(frame)
        finaldict["Number Plate"]= number_plate(frame)
        finaldict["Fire"]= detect_fire(frame)
        print(finaldict)
        output[str("Frame"+str(frame_no))] = finaldict
        print("output",output)
        if(frame_no==10):
            break
    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
   _main()