#import packages
import numpy as np
import pandas as pd
import cv2

#catturo il video ed estraggo le informazioni
cap = cv2.VideoCapture("traffic.mp4")
frames_count, FPS, WIDTH, HEIGHT = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FRAME_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
WIDTH = int(WIDTH)
HEIGHT = int(HEIGHT)
print(frames_count, FPS, WIDTH, HEIGHT)

#creo un pandas DataFrame  con il numero di righe = frame_count
df = pd.DataFrame(index = range(int(frames_count)))
df.index.name = "Frames"

frame_num = 0
cars_crossed_up = 0
cars_crossed_down = 0
car_ids = []
car_ids_crossed = []
total_cars = 0

#creo il background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()
ret, frame = cap.read()
ratio = .5
image = cv2.resize(frame, (0, 0), None, ratio, ratio)
width2, height2, channels = image.shape
#inizializzo il writer
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter("taffic_counter.avi", fourcc, FPS, (height2, width2), True)

while True:
	#import image
	ret, frame = cap.read()
	if ret:
		image = cv2.resize(frame, (0, 0), None, ratio, ratio)
		#converte l'immagine in grigio
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#background subtraction 
		fgmask = fgbg.apply(gray)
		#isola i mezzi dai disturbi per identificarli meglio
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
		closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
		opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
		dilation = cv2.dilate(opening, kernel)
		#rimuove le ombre
		retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)
		#crea i contorni
		im2, contours, hierarchy = cv2.findContours(bins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		#crea i poligoni attorno i contorni
		hull = [cv2.convexHull(c) for c in contours]
		#disegna i contorni
		cv2.drawContours(image, hull, -1, (0, 255, 0), 3)
		lineypos = 255
		cv2.line(image, (0, lineypos), (WIDTH, lineypos), (255, 0, 0), 5)
		lineypos2 = 250
		cv2.line(image, (0, lineypos2), (WIDTH, lineypos2), (0, 255, 0), 5)
		min_area = 300
		max_area = 50000
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
