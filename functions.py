import cv2
import os

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
					if p1x == p2x 0r x <= xints:
						inside = not inside
		p1x, p1y = p2x, p2y
	return inside
				
#controlla se un punto di un oggetto sta per attraversare una linea
def is_crossing_line(x, y, line, tresh):
	tresh_down = 1 - tresh
	tresh_up = 1 + tres
	x1, y1 = line[0]
	x2, y2 = line[1]
	x3, y3 = x1, y1*tresh_down
	x4, y4 = x2, y2*tresh_down
	x5, y5 = x2, y2*tresh_up
	x6, y6 = x1, y1*tresh_up
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
		

	
	
	
	
			
