#import packages
import os
import time
import cv2
import numpy as np
from scipy import *


#controlla se un punto è all'interno di un poligono
def ray_tracing_method(x, y, poly):
	n = len(poly)
	inside = False
	p1x, p1y = poly[0]
	for i in range(n + 1):
		p2x, p2y = poly[i % n]
		if y > min(p1y, p2y):
			if y <= max(p1y, p2y):
				if x <= max(p1x, p2x):
					if p1y != p2y:
						xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
					if p1x == p2x or x <= xints:
						inside = not inside
		p1x, p1y = p2x, p2y
	return inside
				
#controlla se un punto di un oggetto sta per attraversare una linea
def is_crossing_line(x, y, line, thresh):
	thresh_down = 1 - thresh
	thresh_up = 1 + thresh
	x1, y1 = line[0]
	x2, y2 = line[1]
	x3, y3 = x1, y1*thresh_down
	x4, y4 = x2, y2*thresh_down
	x5, y5 = x2, y2*thresh_up
	x6, y6 = x1, y1*thresh_up
	polygon = [(x3, y3), (x4, y4), (x5, y5), (x6, y6)]
	return ray_tracing_method(x, y ,polygon)

#la linea quando passata diventa rossa	
def crossing_color(color, is_crossing):
	if is_crossing:		
		color = 255, 0, 0
		return color
	else:
		return color
		
#funzione che crea i detection box con un pallino al centro
def detectionBox(boxes, frame, IDs, confidence, i, LABELS):
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype = "uint8")
	if len(i) > 0:
		for j in i.flatten():
			#estrae le coordinate del rettangolo
			(x, y) = (boxes[j][0], boxes[j][1])
			(w, h) = (boxes[j][2], boxes[j][3])
			#rettangolo con il testo sopra: nome classe, confidence
			color = [int(c) for c in COLORS[IDs[j]]]
			cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
			text = "{}: {:.4f}".format(LABELS[IDs[j]], confidence[j])
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			#pallino al centro
			cv2.circle(frame, (x + (w//2), y + (h//2)), 2, (0, 255, 0), thickness = 2)
	
#funzione che controlla se il rettangolo di identificazione è gia presente
def BoxGiaPresente(box, detection_attuale, detection_precedente):
	cX, cY, W, H = box
	#distanza minima
	min_dist = np.inf
	frames = 10
	for i in range(frames):
		list_c = list(detection_precedente[i].keys())
		#in caso non ci siano rilevamenti precedenti
		if len(list_c) == 0:
			continue
		temp_dist, index = spatial.KDTree(list_c).query([(cX, cY)])
		if min_dist > temp_dist:
			min_dist = temp_dist
			num_frame = i
			coordinate = list_c[index[0]]
	if min_dist > max(W, H) / 2:
		return False
	#mantengo l'ID
	detection_attuale[(cX, cY)] = detection_precedente[num_frame][coordinate]
	return True

#funzione per pulire temp logs
def clear_logs():
	if os.path.exists(os.path.sep.join([args["inlog"], "In.txt"])):
		os.remove(os.path.sep.join([args["inlog"], "In.txt"]))
	if os.path.exists(os.path.sep.join([args["outlog"], "Out.txt"])):
		os.remove(os.path.sep.join([args["outlog"], "Out.txt"]))

#trova il centro di un rettangolo
def find_center(x, y, w, h):
	x1 = int(w/2)
	y1 = int(h/2)
	cx = x + x1
	cy = y + y1
	return cx, cy
	
	
			
