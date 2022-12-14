import cv2
import os
from functions import *

#classe che gestisce il conteggio dei mezzi
class contatore:
    
    def __init__(self, pos, classi, color, loc):
		#lista di 2 posizioni (x,y) che definiscono la linea
		self.pos = pos
		#lista degli oggetti da contare
		self.classi = classi
		#colore del contatore a schermo
		self.color = color
		#dove piazzare il contatore a schermo nel video
		self.loc = loc
		#valore booleano se qualche oggetto sta passando la linea
		self.is_crossing = False
		#lista di veicoli già contati
		self.veicoli_contati = []
		#dizionario delle varie classi ed il loro conteggio
		self.count_classi = dict(zip(list(classi), [0]*len(classi)))
	
	#metodo che conta i mezzi ed incrementa il contatore relativo
	#veicoli: dizionario con ID del veicolo e le coordinate del centro
	#elements_dict: dizionario con ID del veicolo e la loro classe	
	def count_class(self, veicoli, elements_dict):
		self.is_crossing = False
		x1, y1, x2, y2 = self.pos
		for (ID, center) in veicoli.items():
			if is_crossing_line(center[0], center[1], ((x1, y1), (x2, y2))):
				if ID not in self.veicoli_contati:
					classi = elements_dict[ID]
					if classi in self.count_classi.keys():
						self.count_classi[classi] += 1
					else:
						self.count_classi[classi] = 1
					self.veicoli_contati.append(ID)
					self.is_crossing = True
		return self.count_classi
	
	#metodo che mostra a schermo il contatore
	def count_display(self, img, icons, draw_line = True):
		shape = img.shape[:2]
		offset_r, offset_c = offset_loc(self.draw_loc)
		for n, obj in emnumerate(self.cls):
			r_index_start = offset_r * (icons[obj]["h"] + 20)
			r_index_end = (r_index_start + icons[obj]["h"])
			c_index_start = offset_c * ((icons[obj]["w"] * n) + (40 * (n + 2)))
			c_index_end = (c_index_start + icons[obj]["w"])			
			img[r_index_start : r_index_end, c_index_start : c_index_end] = icons[obj]["icon"]
			cv2.putText(img, "{}".format(self.count_classes[obj]), (abs(min(0, offset_c)*shape[1]) + c_index_end + 5, abs((min(0, offset_r) * img.shape[0])) + r_index_end), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)
		if draw_line:
			cv2.line(img, self.border[:2], self.border[2:], crossing_color(self.color, self.is_crossing), 3)
		
		
