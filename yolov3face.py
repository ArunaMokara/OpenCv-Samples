import cv2
import numpy as np
#import pafy
from utils import*
'''url = 'https://www.youtube.com/watch?v=c07IsbSNqfI&feature=youtu.be'
url="https://www.youtube.com/watch?v=iH1ZJVqJO3Y"
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="mp4")
cap = cv2.VideoCapture(play.url)'''
cap=cv2.VideoCapture("subway.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('frames per second : ' + str(fps))
print('Total number of frames : ' + str(frame_count))
cap.set(3, 480)
cap.set(4, 640)
model_weights="yolov3-wider_16000.weights"
model_cfg="yolov3-face.cfg"
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
def _main():
    while True:
       ret,frame=cap.read()
       blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)
       net.setInput(blob)

    # Runs the forward pass to get output of the output layers
       outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
       faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
       time = cap.get(cv2.CAP_PROP_POS_MSEC)
       time = time / 1000
       frame_no=cap.get(cv2.CAP_PROP_POS_FRAMES)
       print("Face detected at frame:", faces,"Frame number:",frame_no,"Time:", time)
       print('# detected faces: {}'.format(len(faces)))
       print('#' * 10)
       info = [
            ('number of faces detected', '{}'.format(len(faces)))
        ]
       for (i, (txt, val)) in enumerate(info):
            text = '{}: {}'.format(txt, val)
            cv2.putText(frame, text, (10, (i * 20) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
       cv2.imshow("Detection", frame)
       key = cv2.waitKey(1)
       if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break

    cap.release()
    cv2.destroyAllWindows()
    print('==> All done!')
    print('***********************************************************')


if __name__ == '__main__':
    _main()
