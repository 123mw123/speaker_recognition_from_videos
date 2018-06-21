import os
from pytube import YouTube
import cv2


data_sets_location = "C:\\Users\Sai Teja\Desktop\ELL888-CNN"
img_length = 224
img_width = 224


urls = {"SJV":['https://www.youtube.com/watch?v=3J-cYxxHQGQ',
'https://www.youtube.com/watch?v=vQ7ZvPghdy8', 'https://www.youtube.com/watch?v=fTx9tOmU1sY',
'https://www.youtube.com/watch?v=CmVQuiT0OTw', 'https://www.youtube.com/watch?v=nNcFquUuKww',
'https://www.youtube.com/watch?v=rJZjd7rFKws', 'https://www.youtube.com/watch?v=LNyJgNjCDuU',
'https://www.youtube.com/watch?v=e2EPuGabgpc'],
"SM":['https://www.youtube.com/watch?v=oB09kuJa-Eg',
'https://www.youtube.com/watch?v=Y07dnUKwqyw', 'https://www.youtube.com/watch?v=uWnBTyiSwgE',
'https://www.youtube.com/watch?v=b_clFB3vL-Q', 'https://www.youtube.com/watch?v=Ho37w_UFRSg',
'https://www.youtube.com/watch?v=rGpwRlCOLbY'],
"SP":['https://www.youtube.com/watch?v=tXeZupaycWI', 'https://www.youtube.com/watch?v=sSTOxl2J1WI',
'https://www.youtube.com/watch?v=dkbtHVayA3U', 'https://www.youtube.com/watch?v=crqlI1Exte4',
'https://www.youtube.com/watch?v=klP7mKwhweo', 'https://www.youtube.com/watch?v=4eiyui0dF5o'],
"AK":['https://www.youtube.com/watch?v=x09Ft-XdChg','https://www.youtube.com/watch?v=Lc81vSTHGMY',
      'https://www.youtube.com/watch?v=udQ4IgfFzvo','https://www.youtube.com/watch?v=CHQ3odjT7oY'],
"SK":['https://www.youtube.com/watch?v=Wp_Al0AYylA&list=PLzdjxgz3O7oMZLoZsqAwX_',
      'https://www.youtube.com/watch?v=xkX6RU-q4N0&list=PLzdjxgz3O7oMZLoZsqAwX_4lWRLouhZub&index=',
      'https://www.youtube.com/watch?v=t2P9fDEKCVE&index=14&list=PLzdjxgz3O7oMZLoZsqAwX_4lWRLouhZub',
      'https://www.youtube.com/watch?v=4CCqAdsHl-8&index=19&list=PLzdjxgz3O7oMZLoZsqAwX_4lWRLouhZub',
      'https://www.youtube.com/watch?v=4o8E5dyvYCk&index=17&list=PLzdjxgz3O7oMZLoZsqAwX_4lWRLouhZub'],
 "FR":['https://www.youtube.com/watch?v=BTvFi5SZJnw', 'https://www.youtube.com/watch?v=Qp1Vj_-Kg3M',
       'https://www.youtube.com/watch?v=ir8o5Fxn4yk', 'https://www.youtube.com/watch?v=_dXZ_dywfa4',
        'https://www.youtube.com/watch?v=I3I-ab9FHHA', 'https://www.youtube.com/watch?v=_olkwh6lQ_s']
        }



for p in urls.keys():
    try:
        os.makedirs(data_sets_location + "\\" + p)
        os.makedirs(data_sets_location + "\\" + p +"_videos")
    except FileExistsError:
        pass

    for url in urls[p]:
        yt = YouTube(url)
        stream = yt.streams.get_by_itag(43) #getting 360p resolution video
        print(stream)

        stream.download(data_sets_location + "\\" + p +"_videos")


    count = 1
    os.chdir(data_sets_location + "\\" + p)
    for video in os.listdir(data_sets_location + "\\" + p + "_videos"):
        print(video)
        vidcap = cv2.VideoCapture(data_sets_location + "\\" + p + "_videos"+"\\"+video)
        success, image = vidcap.read()
        success = True
        while success:
            success, image = vidcap.read()
            if success:
                
                image = cv2.resize(image, (224, 224))

                cv2.imwrite(p+"_%d.jpg" % count, image)  # save frame as JPEG file
                count += 1


