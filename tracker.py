#import packages
import numpy as np
from collections import *
from scipy.spatial import *

class Tracker():
	def __init__(self, maxDisappeared = 50):
		#contatore usato per assegnare Id unici agli oggetti
		self.nextID = 0
		#dizionario che usa ID come chiave e le coordinate del centro come valore
		self.objects = OrderedDict()
		#mantiene il numero di frame in cui un ID is missing
		self.disappeared = OrderedDict()
		#il numero massimo di frame consecutivi per cui un oggetto poi va rimosso
		self.maxDisappeared = maxDisappeared
	 
	def register(self, center):
		self.objects[self.nextID] = center
		self.disappeared[self.nextID] = 0
		self.nextID += 1
	
	def deregister(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]
	
	def update(self, boxes):
		#controllo se la lista dei bounding box è vuota
		if len(boxes) == 0:
			# controllo ogni oggetto esistente e lo marchio come scomparso
			for ID in self.disappeared.keys():
				self.disappeared[ID] += 1
				#se ho raggiunto il massimo numero di frame lo deregistro
				if self.disappeared[ID] > self.maxDisappeared:
					self.deregister(ID)
			return self.objects
		#inizializzo un array di centri dei bounding box
		input_centroids = np.zeros((len(boxes), 2), dtype = "int")
		#estraggo le coordinate di tutti i bounding box e da quelle trovo il centro
		for (i, (x1, y1, x2, y2)) in enumerate(boxes):
			cx = int((x1 + x2) / 2.0)
			cy = int((y1 + y2) / 2.0)
			input_centroids[i] = (cx, cy)
		#in caso non ci sia nessun elemento "marchiato" prendo i centri in input e li registro
		if len(self.objects) == 0:
			for x in range(0, len(input_centroids)):
				self.register(input_centroids[x])
		#caso contrario ci sono elementi già presenti
		else:
			#estraggo gli ID e i centri dei boxes presenti
			IDs = list(self.objects.keys())
			object_centroids = list(self.objects.values())
			#calcolo la distanza tra i centri in input e quelli presenti
			D = dist.cdist(np.array(object_centroids), input_centroids)
			rows = D.min(axis = 1).argsort()
			cols = D.argmin(axis = 1)[rows]
			usedRows = set()
			usedCols = set()
			for (r, c) in zip(rows, cols):
				if r in usedRows or c in usedCols:
					continue
				ID = IDs[row]
				self.objects[ID] = input_centroids[c]
				self.disappeared[ID] = 0
				usedRows.add(r)
				usedCols.add(c)
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)
			if D.shape[0] >= D.shape[1]:
				for r in unusedRows:
					ID =IDs[r]
					self.disappeared[ID] += 1
					if self.disappeared[ID] > self.maxDisappeared:
						self.deregister(ID)
			else:
				for c in unusedCols:
					self.register(input_centroids[c])
		return self.objects
			
			
			
			
			
			
			
			
			
			
			
			
			
			
