import numpy as np
import imutils
import time as t
import requests, json , os
from elasticsearch import Elasticsearch
import json
import cv2
from demog import*
from no_plate_detection_video import*
from yolo_edit import*
from yolo_video import*
from Fire_detection_video import*
from obj_video import*
import pymongo


'''connection = pymongo.MongoClient('localhost:27017')
database = connection.get_database('demotesting')
data = database.get_collection(('response'))
cursor = data.find()
path=[]

for record in cursor:
    if record['processStatus']=='pending':
        path.append(record['fileDownloadUri'])

print(path)'''
path=['/home/administrator/PycharmProjects/Internship/Intern/YOLOv3/combine/combine.mp4','/home/administrator/PycharmProjects/Internship/Intern/YOLOv3/combine/output_check.mp4']
res = requests.get('http://localhost:9200')
print (res.content)
es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
k = 0
queue=[]
for i in path:
    if i not in queue:
        queue.append(i)
for url in range(len(path)):
    print("True")
    cap = cv2.VideoCapture(queue[0])
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('frames per second : ' + str(fps))
    print('Total number of frames : ' + str(frame_count))
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            #print("Done")
            # queue.pop(0)
            break
        time = cap.get(cv2.CAP_PROP_POS_MSEC)
        time = round(time / 1000, 2)
        frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
        finaldict = {}
        paths=queue[0]
        finaldict["video_url"]=paths
        finaldict["video_id"] = 1
        for j in range(len(paths)):
            if(paths[j]=="."):
                finaldict["video_type"]=paths[j:]
        finaldict["Frame_no"] = int(frame_no)
        finaldict["Time"]= time
        finaldict["Detections"]={}
        finaldict["Detections"]["Person_detections"]=persons(frame)
        finaldict["Detections"]["Demographic_detections"]=demographics(frame)
        finaldict["Detections"]["Color_detections"]=color_detection(frame)
        finaldict["Detections"]["Animal_detections"]=animal(frame)
        finaldict["Detections"]["Numberplate_detections"] = number_plate(frame)
        finaldict["Detections"]["Fire_detections"] = detect_fire(frame)
        print(finaldict)
        i = i + 1
        with open('video'+str(k+1)+'/'+str(i)+'.json', 'w', encoding='utf-8') as f:
            json.dump(finaldict, f, ensure_ascii=False, indent=4)

        if(frame_no==5):
            print("Done")
            queue.pop(0)
            break
    link='video'+str(k+1)+'/'
    for filename in os.listdir(link):
        if filename.endswith(".json"):
            f = open('video'+str(k+1)+'/'+filename)
            docket_content = f.read()
            #print(docket_content)
            ind = 'demo'+str(k+1)
            es.index(index=ind, ignore=400, doc_type='doc', body=json.loads(docket_content))
    k = k + 1

