import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import uuid
import os

def color_dominant(x, y, w, h, frame):
    def find_histogram(clt):
        """
        create a histogram with k clusters
        :param: clt
        :return:hist
        """
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)

        hist = hist.astype("float")
        hist /= hist.sum()

        return hist

    #image = cv2.imread("human.jpg")
    def plot_colors2(hist, centroids):
        bar = np.zeros((50, 300, 3), dtype="uint8")
        startX = 0
        percent_list= []
        color_list = []
        value = []
        Dict={}
        for (percent, color) in zip(hist, centroids):
            #print("########################")
            percent_list.append(percent)
            color_list.append(list(color))
            # plot the relative percentage of each cluster
            endX = startX + (percent * 300)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                          color.astype("uint8").tolist(), -1)
            startX = endX

        for i in percent_list, color_list:
            #Dict = dict({"percentage":percent_list, "color": color_list})
            Dict["percentage"]=percent_list
            Dict["color"]=color_list
        value.append(max(percent_list))
        #print("maximum percentage: ", value)
        return bar, value,Dict
        # return the bar chart
    #for item in coordinates:
        #(X, Y, W, H) = item
    directory = "ROI Images"
    # Parent Directory path
    parent_dir = os.getcwd()
    # Path
    path = os.path.join(parent_dir, directory)
    try:
        os.mkdir(path)
    except:
        print()
    finally:
        print()
    roi = frame[y:y + h, x:x + w]
    a = uuid.uuid4()
    #print("roi",roi)
    #print(roi.shape)
    #print("roi length",roi, len(roi), roi.shape[1])
    if roi.shape[1] == 0 or roi.shape[0] ==0 or roi.shape[2] ==0:
        return None
    cv2.imwrite(os.path.join(path , "{0}.jpg".format(a)), roi)
    pict = cv2.imread(os.path.join(path, "{0}.jpg".format(a)))
    img1 = cv2.cvtColor(pict, cv2.COLOR_BGR2RGB)
    img1 = img1.reshape((img1.shape[0] * img1.shape[1],3))#represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img1)
    hist = find_histogram(clt)
    bar, value,dict= plot_colors2(hist, clt.cluster_centers_)
    return dict
    #plt.axis("off")
    #plt.imshow(bar)
    #plt.show()'''

