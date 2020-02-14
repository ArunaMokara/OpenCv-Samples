import cv2
import numpy as np
from utils import*
model_weights="yolov3-wider_16000.weights"
model_cfg="yolov3-face.cfg"
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
dict = {}
def demographics(frame):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))
         # Remove the bounding boxes with low confidence
        faces, genders, ages, confidences = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        male_count = 0
        female_count = 0
        no_of_faces = len(faces)
        dict["Faces found"] = no_of_faces
        dict["Demographics"] = {}
        for i in range(len(faces)):
            dict["Demographics"][str("person "+str(i+1))] = {}
            dict["Demographics"][str("person "+str(i+1))]["gender"] = genders[i]
            if(genders[i]=="Male"):
                male_count+=1
            else:
                female_count+=1
            dict["Demographics"][str("person "+str(i+1))][" age_range"] = ages[i]
            dict["Demographics"][str("person "+str(i+1))]["Confidence:"] = round(confidences[i], 2)
        dict["Males:"] = male_count
        dict["Females:"] = female_count

        info = [
                ('number of faces detected', '{}'.format(len(faces)))
            ]
        for (i, (txt, val)) in enumerate(info):
            text = '{}: {}'.format(txt, val)
            cv2.putText(frame, text, (10, (i * 20) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
        return dict
