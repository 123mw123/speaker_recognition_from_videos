import os
import cv2
from PIL import Image

videos_location = "C:\\Users\Sai Teja\Desktop\ELL888-CNN\\videos"
DATA_DIR = "C:\\Users\Sai Teja\Desktop\ELL888-CNN\small"


try:
    os.makedirs(DATA_DIR )
except FileExistsError:
    pass

try:
    os.makedirs(DATA_DIR + "\\" + "train")
    os.makedirs(DATA_DIR + "\\" + "test")
    os.makedirs(DATA_DIR + "\\" + "val")
except FileExistsError:
    pass

try:
    os.makedirs(DATA_DIR + "\\" + "train" + "\\" + "SGV")
    os.makedirs(DATA_DIR + "\\" + "train" + "\\" + "SM")
    os.makedirs(DATA_DIR + "\\" + "train" + "\\" + "SP")
    os.makedirs(DATA_DIR + "\\" + "train" + "\\" + "AK")
    os.makedirs(DATA_DIR + "\\" + "train" + "\\" + "FR")
    os.makedirs(DATA_DIR + "\\" + "train" + "\\" + "SK")

    os.makedirs(DATA_DIR + "\\" + "test" + "\\" + "SGV")
    os.makedirs(DATA_DIR + "\\" + "test" + "\\" + "SM")
    os.makedirs(DATA_DIR + "\\" + "test" + "\\" + "SP")
    os.makedirs(DATA_DIR + "\\" + "test" + "\\" + "AK")
    os.makedirs(DATA_DIR + "\\" + "test" + "\\" + "FR")
    os.makedirs(DATA_DIR + "\\" + "test" + "\\" + "SK")

    os.makedirs(DATA_DIR + "\\" + "val" + "\\" + "SGV")
    os.makedirs(DATA_DIR + "\\" + "val" + "\\" + "SM")
    os.makedirs(DATA_DIR + "\\" + "val" + "\\" + "SP")
    os.makedirs(DATA_DIR + "\\" + "val" + "\\" + "AK")
    os.makedirs(DATA_DIR + "\\" + "val" + "\\" + "FR")
    os.makedirs(DATA_DIR + "\\" + "val" + "\\" + "SK")

except FileExistsError:
    pass

persons = ["SGV","SM","SP","AK","FR","SK"]

for p in persons:
    count = 0
    v = 0
    for video in os.listdir(videos_location + "\\" + p + "_videos"):
        if(v<4):
            v=v+1
            os.makedirs(DATA_DIR + "\\" + "train" + "\\" + p+"\\"+str(v))
            os.makedirs(DATA_DIR + "\\" + "test" + "\\" + p+"\\"+str(v))
            os.makedirs(DATA_DIR + "\\" + "val" + "\\" + p+"\\"+str(v))

            print(video.encode("utf-8"))
            vidcap = cv2.VideoCapture(videos_location + "\\" + p + "_videos"+"\\"+video)
            success, image = vidcap.read()
            success = True
            length = 0
            while success:
                success, image = vidcap.read()
                if success:
                    length = length+1
                    
            trn = int(length/600)
            tst = int(length/75)
            vld = int(length/75)+1
            vidcap = cv2.VideoCapture(videos_location + "\\" + p + "_videos"+"\\"+video)
            success, image = vidcap.read()
            success = True

            while success:

                success, image = vidcap.read()
                if success:
                    if count%tst == 0:

                        os.chdir(DATA_DIR + "\\" + "test" + "\\" + p+"\\"+str(v))
                        image = cv2.resize(image,(224,224), interpolation = cv2.INTER_CUBIC)
                        cv2.imwrite(p+"_%d.jpg" % count, image)  # save frame as JPEG file
                    elif count%vld == 0:
                        os.chdir(DATA_DIR + "\\" + "val" + "\\" + p+"\\"+str(v))
                        image = cv2.resize(image,(224,224), interpolation = cv2.INTER_CUBIC)
                        cv2.imwrite(p+"_%d.jpg" % count, image)
                    elif count%trn == 0:
                        os.chdir(DATA_DIR + "\\" + "train" + "\\" + p+"\\"+str(v))
                        image = cv2.resize(image,(224,224), interpolation = cv2.INTER_CUBIC)
                        cv2.imwrite(p+"_%d.jpg" % count, image)

                    count += 1

