import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

pics_location = "C:\\Users\Sai Teja\Desktop\ELL888-CNN\\small"
DATA_DIR = "C:\\Users\Sai Teja\Desktop\ELL888-CNN\\faces_large"


try:
    os.makedirs(DATA_DIR )
except FileExistsError:
    pass

try:
    os.makedirs(DATA_DIR + "\\" + "train")
    os.makedirs(DATA_DIR + "\\" + "test")
    
except FileExistsError:
    pass

try:
    os.makedirs(DATA_DIR + "\\" + "train" + "\\" + "SGV")
    os.makedirs(DATA_DIR + "\\" + "train" + "\\" + "SM")
    os.makedirs(DATA_DIR + "\\" + "train" + "\\" + "SP")
    os.makedirs(DATA_DIR + "\\" + "train" + "\\" + "AK")
    os.makedirs(DATA_DIR + "\\" + "train" + "\\" + "FR")
    os.makedirs(DATA_DIR + "\\" + "train" + "\\" + "SK")
    os.makedirs(DATA_DIR + "\\" + "train" + "\\" + "Noise")

    os.makedirs(DATA_DIR + "\\" + "test" + "\\" + "SGV")
    os.makedirs(DATA_DIR + "\\" + "test" + "\\" + "SM")
    os.makedirs(DATA_DIR + "\\" + "test" + "\\" + "SP")
    os.makedirs(DATA_DIR + "\\" + "test" + "\\" + "AK")
    os.makedirs(DATA_DIR + "\\" + "test" + "\\" + "FR")
    os.makedirs(DATA_DIR + "\\" + "test" + "\\" + "SK")
    os.makedirs(DATA_DIR + "\\" + "test" + "\\" + "Noise")


except FileExistsError:
    pass


 
DIR =  "C:\\Users\Sai Teja\Desktop\ELL888-CNN\deep-learning-face-detection"

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(DIR+"\\deploy.prototxt.txt", DIR+"\\facedetect.caffemodel")


persons = ["SGV","SM","SP","AK","FR","SK",'Noise']

for p in persons:
    count =0
    for pic in os.listdir(pics_location+'\\'+'test'+'\\'+p):
        count= count+1
        image = cv2.imread(pics_location+'\\'+'test'+'\\'+p+"\\"+pic)
        os.chdir(DATA_DIR + "\\" + "train" + "\\" + p)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()
        print(detections.shape)
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
         
                
                y = startY - 10 if startY - 10 > 10 else startY + 10
                #cv2.rectangle(image, (startX, startY), (endX, endY),
                 #   (0, 0, 255), 2)
                crpim =  image[startY-20:endY+20, startX-20:endX+20]
                if(len(crpim)!=0):
                    try:
                        crpim = cv2.resize(crpim,(96,96), interpolation = cv2.INTER_CUBIC)
                        cv2.imwrite(str(count)+'.jpg',crpim)
                    except Exception as e:
                        pass

    for pic in os.listdir(pics_location+'\\'+'val'+'\\'+p):
        count= count+1
        image = cv2.imread(pics_location+'\\'+'val'+'\\'+p+"\\"+pic)
        os.chdir(DATA_DIR + "\\" + "test" + "\\" + p)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()
        print(detections.shape)
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
         
                # draw the bounding box of the face along with the associated
                # probability
                #text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                #cv2.rectangle(image, (startX, startY), (endX, endY),
                 #   (0, 0, 255), 2)
                crpim =  image[startY-20:endY+20, startX-20:endX+20]
                if(len(crpim)!=0):
                    try:
                        crpim = cv2.resize(crpim,(96,96), interpolation = cv2.INTER_CUBIC)
                        cv2.imwrite(str(count)+'.jpg',crpim)
                    except Exception as e:
                        pass
                