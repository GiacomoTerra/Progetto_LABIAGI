import cv2
import os
import numpy as np

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
	thresh_up = 1 + thres
	x1, y1 = line[0]
	x2, y2 = line[1]
	x3, y3 = x1, y1*thresh_down
	x4, y4 = x2, y2*thresh_down
	x5, y5 = x2, y2*thresh_up
	x6, y6 = x1, y1*thresh_up
	polygon = [(x3, y3), (x4, y4), (x5, y5), (x6, y6)]
	res = rey_tracing_method(x, y, polygon)
	return res

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
	
	
	
	
			
