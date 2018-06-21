import imageio
imageio.plugins.ffmpeg.download()
import moviepy.editor as mp
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from sklearn.preprocessing import MinMaxScaler
import cv2
import os
import time
import subprocess
 
# Wait for 5 seconds



videos_location = "C:\\Users\Sai Teja\Desktop\ELL888-CNN\\video"
DATA_DIR = "C:\\Users\Sai Teja\Desktop\ELL888-CNN\\Spectrograms\\data_mfcc"


try:
	os.makedirs(DATA_DIR )
except FileExistsError:
	pass
try:
	os.makedirs(DATA_DIR  + "\\" +"train"+"\\" + "SGV")
	os.makedirs(DATA_DIR  + "\\" +"train"+"\\"+ "SM")
	os.makedirs(DATA_DIR  + "\\" +"train"+"\\" + "SP")
	os.makedirs(DATA_DIR  + "\\" +"train"+"\\" + "AK")
	os.makedirs(DATA_DIR  + "\\" +"train"+"\\" + "FR")
	os.makedirs(DATA_DIR  + "\\" +"train"+"\\" + "SK")
	os.makedirs(DATA_DIR  + "\\" +"val"+"\\" + "SGV")
	os.makedirs(DATA_DIR  + "\\" +"val"+"\\" + "SM")
	os.makedirs(DATA_DIR  + "\\" +"val"+"\\" + "SP")
	os.makedirs(DATA_DIR  + "\\" +"val"+"\\" + "AK")
	os.makedirs(DATA_DIR  + "\\" +"val"+"\\" + "FR")
	os.makedirs(DATA_DIR  + "\\" +"val"+"\\" + "SK")


except FileExistsError:
	pass

persons = ["AK","FR","SGV","SK","SM","SP"]
for p in persons:
	v = 0
	count = 0
	for video in os.listdir(videos_location):
		print(video.encode("utf-8"))
		
		vid = mp.VideoFileClip(videos_location +"\\"+video)
		d = vid.duration
		print(d)
		clip =vid.subclip(0,d)
		clip.audio.write_audiofile("C:\\Users\Sai Teja\Desktop\ELL888-CNN\\Spectrograms"+"\\"+p+str(d)+".wav")
		subprocess.call(['ffmpeg', '-i',"C:\\Users\Sai Teja\Desktop\ELL888-CNN\\Spectrograms"+"\\"+p+str(d)+".wav",
			  '-af', "highpass=f=200, lowpass=f=3000", "C:\\Users\Sai Teja\Desktop\ELL888-CNN\\Spectrograms"+"\\"+p+str(d)+"denoise_audio.wav"])
		clip.reader.close()
		audio = mp.AudioFileClip( "C:\\Users\Sai Teja\Desktop\ELL888-CNN\\Spectrograms"+"\\"+p+str(d)+"denoise_audio.wav")
		d = audio.duration
		print(d)
		i =20
		while( i<=d-25 and count<394 and i<420 ):

			clip =audio.subclip(i,i+5)
			clip.write_audiofile("C:\\Users\Sai Teja\Desktop\ELL888-CNN\\Spectrograms"+"\\"+"theaudio.wav")
			#clip.reader.close()
			
			i=i+5
			print(i)
			


			y, sr = librosa.load("C:\\Users\Sai Teja\Desktop\ELL888-CNN\\Spectrograms"+"\\"+"theaudio.wav")
			S = librosa.feature.mfcc(y, sr=sr,  n_mfcc=20)

			# Convert to log scale (dB). We'll use the peak power as reference.
			log_S = librosa.power_to_db(S, ref=np.max)

			scaler = MinMaxScaler(feature_range=(0, 255))
			count = count+1
			img = scaler.fit_transform(log_S)
			if(count%10==0):
				os.chdir(DATA_DIR  +  "\\" +"val"+"\\" + p)
				img = cv2.resize(img,(64,64))
				cv2.imwrite(str(count)+'.jpg',img)
			else:
				os.chdir(DATA_DIR  +  "\\" +"train"+"\\" + p)
				img = cv2.resize(img,(64,64))
				cv2.imwrite(str(count)+'.jpg',img)
	print(count)
			