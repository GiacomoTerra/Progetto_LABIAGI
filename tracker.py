#import packages
import numpy as np
import math

class Tracker():
	def __init__(self):
		#memorizza il centro di un oggetto
		self.center = {}
		#contatore degli IDs
		self.counter = 0
	
	def update(self, boxes):
		#boxes e IDs
		bxs_ids = []
		#prendo il centro del nuovo oggetto
		for box in boxes:
			(x, y, w, h, index) = box
			cx = (x + x + w) // 2
			cy = (y + y + h) // 2
			#in caso l'oggetto sia già detected
			object_detected = False
			for id, pt in self.center.items():
				dist = math.hypot(cx - pt[0], cy - pt[1])
				if dist < 25:
					self.center[id] = (cx, cy)
					bxs_ids.append([x, y, w, h, id, index])
					object_detected = True
					break
			#nuovo oggetto nuovo ID
			if object_detected is False:
				self.center[self.counter] = (cx, cy)
				bxs_ids.append([x, y, w, h, self.counter, index])
				self.counter += 1
		#rimuovo gli IDs non piu utilizzati
		new_center = {}
		for bx_id in bxs_ids:
			_,_,_,_, object_id, index = bx_id
			c = self.center_points[object_id]
			new_center[object_id] = c
		self.center = new_center.copy()
		return bxs_ids
		
			
			
			
			
			
			
			
			
			
			
