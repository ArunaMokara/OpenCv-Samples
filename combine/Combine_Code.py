#importing necessary packages and individual codes
import cv2
from demog import*
from no_plate_detection_video import*
from yolo_edit import*
from yolo_video import*
from Fire_detection_video import*
from obj_video import*
import pymongo
from bson.objectid import ObjectId
import requests, json
from elasticsearch import Elasticsearch
import datetime
import time as t

#Connecting to elasticsearch
requests.get('http://localhost:9200')
elasticsearch = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
print("outside")
#function for processing video in all codes
def processing():

    print("In processing ", str(datetime.datetime.now()))

    #connecting to mongodb and getting collection
    connection = pymongo.MongoClient('localhost:27017')
    database = connection.get_database('demotesting')
    upload_collection = database.get_collection(('response'))

    #creating a cursor for collection
    upload_cursor = upload_collection.find()

    #creating lists to store userids, urls, videoids
    videoIds=[]
    urls=[]
    userIds=[]

    #looping through cursor to get pending videos
    for record in upload_cursor:
        if record['processStatus'] == 'pending' or record['processStatus'] == 'In_Progress':
            videoIds.append(str(record['_id']))
            urls.append(record['fileUrl'])
            userIds.append(record['userid'])

    #creating a queue to store urls to process
    queue=[]

    #looping through urls to remove processed videos
    for i in urls:
        if i not in queue:
            queue.append(i)

    #looping through urls to process each video
    for url in range(len(urls)):

        #reading video
        cap = cv2.VideoCapture(queue[0])

        #updating processStatus of video as In_Progress
        upload_collection.find_one_and_update(
            {"_id": ObjectId(videoIds[0])},
            {"$set":
                 {"processStatus": "In_Progress"}
             }, upsert=False
        )

        #printing fps and total frames to console
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('frames per second : ' + str(fps))
        print('Total number of frames : ' + str(frame_count))

        #looping through each frame in the video
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Done")
                queue.pop(0)
                break
            time = cap.get(cv2.CAP_PROP_POS_MSEC)
            time = round(time / 1000, 2)
            time = t.strftime('%H:%M:%S', t.gmtime(time))
            frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)

            #creating and storing each frame detections  in a dictionary
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

            #getting detections from each individual code
            finaldict["Detections"]["Person_detections"]=persons(frame)
            finaldict["Detections"]["Demographic_detections"]=demographics(frame)
            finaldict["Detections"]["Color_detections"]=color_detection(frame)
            finaldict["Detections"]["Animal_detections"]=animal(frame)
            finaldict["Detections"]["Numberplate_detections"] = number_plate(frame)
            finaldict["Detections"]["Fire_detections"] = detect_fire(frame)

            #creating index as userId_videoId
            index =userIds[0]+'_'+ videoIds[0]#'video' + str(k + 1)
            finaldict = json.dumps(finaldict)
            print(finaldict)

            #inserting into elasticsearch
            elasticsearch.index(index=index, ignore=400, doc_type='doc', body=finaldict)
            print("index",index)
            # if(frame_no==5):
            #     print("Done")
            #     queue.pop(0)
            #     break

        #updating processStatus of video as completed
        upload_collection.find_one_and_update(
            {"_id": ObjectId(videoIds[0])},
            {"$set":
                 {"processStatus": "completed"}
             }, upsert=False
        )
        videoIds.pop(0)
        userIds.pop(0)
