#import packages
import numpy as np
import pandas as pd
import cv2

#catturo il video ed estraggo le informazioni
cap = cv2.VideoCapture("inputVideos/highway.mp4")
frames_count, FPS, WIDTH, HEIGHT = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
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
image = cv2.resize(frame, (WIDTH, HEIGHT), None, ratio, ratio)
width2, height2, channels = image.shape
#inizializzo il writer
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter("highway.avi", fourcc, FPS, (height2, width2), True)

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
		contours, hierarchy = cv2.findContours(bins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		#crea i poligoni attorno i contorni
		hull = [cv2.convexHull(c) for c in contours]
		#disegna i contorni
		cv2.drawContours(image, hull, -1, (0, 255, 0), 3)
		lineypos = 225
		cv2.line(image, (0, lineypos), (WIDTH, lineypos), (255, 0, 0), 5)
		lineypos2 = 250
		cv2.line(image, (0, lineypos2), (WIDTH, lineypos2), (0, 255, 0), 5)
		min_area = 500
		max_area = 50000
		#vettori per i centri dei contorni individuati
		cxx = np.zeros(len(contours))
		cyy = np.zeros(len(contours))
		#ciclo su tutti i contorni presenti sul frame
		for i in range(len(contours)):
			#conto solamente i contorni principali
			if hierarchy[0, i, 3] == -1:
				area = cv2.contourArea(contours[i])
				#area nell'intervallo prestabilito
				if min_area < area < max_area:
					# calculating centroids of contours
					cnt = contours[i]
					M = cv2.moments(cnt)
					cx = int(M['m10'] / M['m00'])
					cy = int(M['m01'] / M['m00'])
					if cy > lineypos:
						#prendo le coordinate del rettangolo
						x, y, w, h = cv2.boundingRect(cnt)
						#creo il rettangolo
						cv2.rectangle(image, (x,y), (x + w, y + h), (255, 0, 0), 2)
						cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 1)
						cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1, line_type=cv2.LINE_AA)
						#aggiungo i centri ai vettori 
						cxx[i] = cx
						cyy[i] = cy
		#elimino le entries nulle centri che non sono stati aggiunti
		cxx = cxx[cxx != 0]
		cyy = cyy[cyy != 0]
		#liste per controllare quali indici sono stati aggiunti al data frame
		minx_index2 = []
		miny_index2 = []
		maxrad = 25
		#almeno un'identificazione
		if len(cxx):			
			#nessun id presente
			if not car_ids:
				for i in range(len(cxx)):
					car_ids.append(i)
					#aggiunge una colonna al data frame relativa al id
					df[str(car_ids[i])] = ""
					#assegna il centro al frame e car id corrente
					df.at[int(frame_num), str(car_ids[i])] = [cxx[i], cyy[i]]
					total_cars = car_ids[i] + 1
			else:
				#array per calcolare i delta
				dx = np.zeros((len(cxx), len(car_ids)))
				dy = np.zeros((len(cyy), len(car_ids)))
				for i in range(len(cxx)):
					#ciclo tra gli id presenti
					for j in range(len(car_ids)):
						#prendo il centro dal frame precedente
						old_cx_cy = df.iloc[int(frame_num - 1)][str(car_ids[j])]
						#centro del frame attuale
						current_cx_cy = np.array([cxx[i], cyy[i]])
						#in caso il vecchio centro sia empty
						if not old_cx_cy:
							#continua al prossimo id
							continue
						#calcolo la differenza
						else:
							dx[i, j] = old_cx_cy[0] - current_cx_cy[0]
							dy[i, j] = old_cx_cy[0] - current_cx_cy[0]
				for j in range(len(car_ids)):
					#sommo i delta
					somma = np.abs(dx[:, j]) + np.abs(dy[:, j])
					#l'id che ha la minore differenza è quello vero
					true_index = np.argmin(np.abs(somma))
					minx_index = true_index
					miny_index = true_index
					#salvo i valori per controllare successivamente il raggio minimo
					min_dx = dx[minx_index, j]
					min_dy = dy[miny_index, j]
					#controllo se il minimo ed i delta sono nulli(se il centro non si è mosso)
					if min_dx == 0 and min_dy == 0 and np.all(dx[:, j] == 0) and np.all(dy[:, j] == 0):
						#continua al prossimo id
						continue
					else:
						#i delta sono minori del raggio max
						if np.abs(min_dx) < maxrad and np.abs(min_dy) < maxrad:
							#aggiunge il centro all'id già esistente
							df.at[int(frame_num), str(car_ids[j])] = [cxx[minx_index], cyy[miny_index]]
							#aggiungo gli indici
							minx_index2.append(minx_index)
							miny_index2.append(miny_index)
				for i in range(len(cxx)):
					#centro non presente nella lista
					if i not in minx_index2 and miny_index2:
						#nuova colonna con il totale
						df[str(total_cars)] = ""
						total_cars += 1
						t = total_cars - 1
						car_ids.append(t)
						df.at[int(frame_num), str(t)] = [cxx[i], cyy[i]]
					elif current_cx_cy[0] and not old_cx_cy and not minx_index2 and not miny_index2:
						df[str(total_cars)] = ""
						total_cars += 1
						t = total_cars - 1
						car_ids.append(t)
						df.at[int(frame_num), str(t)] = [cxx[i], cyy[i]]
		
		#macchine su schermo
		current_cars = 0
		#indice macchine a schermo
		current_cars_index = []
		for i in range(len(car_ids)):
			#controllo il data frame per vedere quali ids sono attivi nel frame
			if df.at[int(frame_num), str(car_ids[i])] != '':
				#aggiunge una macchina al totale
				current_cars += 1
				#aggiunge l'indice
				current_cars_index.append(i)
		for i in range(current_cars):
			#prende un centro di un id del frame corrente
			current_center = df.iloc[int(frame_num)][str(car_ids[current_cars_index[i]])]
			#stessa cosa frame precedente
			old_center = df.iloc[int(frame_num - 1)][str(car_ids[current_cars_index[i]])]
			#se esiste un centro
			if current_center:
				cv2.putText(image, "Centroid" + str(current_center[0]) + "," + str(current_center[1]), (int(current_center[0]), int(current_center[1])), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 255, 255), 2)
				cv2.putText(image, "ID:" +str(car_ids[current_cars_index[i]]), (int(current_center[0]), int(current_center[1] - 15)), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 0xFF, 0xFF), 2)
				cv2.drawMarker(image, (int(current_center[0]), int(current_center[1])), (0, 0, 0xFF), cv2.MARKER_STAR, markerSize = 5, thickness = 1 , line_type = cv2.LINE_AA)
				#controlla se esistono vecchi centri
				if old_center:
					#visualizza il box
					x_start = old_center[0] - maxrad
					y_start = old_center[1] - maxrad
					x_width = old_center[0] + maxrad
					y_height = old_center[1] + maxrad
					cv2.rectangle(image, (int(x_start), int(y_start)), (int(x_width), int(y_height)), (0, 125, 0), 1)
					#controlla se il vecchio centroide è sopra o sotto la linea e quello corrente è sopra o oltre
					if old_center[1] >= lineypos2 and current_center[1] <= lineypos2 and car_ids[current_cars_index[i]] not in car_ids_crossed:
						#incremento il contatore su
						cars_crossed_up = cars_crossed_up + 1
						cv2.line(image, (0, lineypos2), (WIDTH, lineypos2), (0, 0, 255), 5)
						#aggiunge l'id alla lista dei mezzi contati
						car_ids_crossed.append(current_cars_index[i])
					#controlla se il vecchio centroide è sopra o oltre la linea e quello corrente è prima della linea
					elif old_center[1] <= lineypos2 and current_center[1] >= lineypos2 and car_ids[current_cars_index[i]] not in car_ids_crossed:
						cars_crossed_down = cars_crossed_down + 1
						cv2.line(image, (0, lineypos2), (WIDTH, lineypos2),	(0, 0, 125), 5)
						car_ids_crossed.append(current_cars_index[i])

		
		
		#rettangolo in background
		cv2.rectangle(image, (0, 0), (250, 100), (255, 0, 0), -1)
		#tutti i contatori a schermo
		cv2.putText(image, "Cars in Area: " + str(current_cars), (0, 15), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 170, 0), 1)
		cv2.putText(image, "Cars Crossed Up: " + str(cars_crossed_up), (0, 30), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 170, 0), 1)
		cv2.putText(image, "Cars Crossed Down: " + str(cars_crossed_down), (0, 45), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 170, 0), 1)
		cv2.putText(image, "Total Cars Detected: " + str(len(car_ids)), (0, 60), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 170, 0), 1)
		cv2.putText(image, "Frame: " + str(frame_num) + " of " + str(frames_count), (0, 75), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 170, 0), 1)
		cv2.putText(image, "Time: " + str(round(frame_num / FPS, 2)) + " sec of " + str(round(frames_count / FPS, 2)) + " sec ", (0, 90), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 170, 0), 1)
		
		#displays images e transformations
		cv2.imshow("countours", image)
		cv2.moveWindow("countours", 0, 0)
		
		
		
		writer.write(image)
		frame_num += 1
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break
cap.release()
cv2.destroyAllWindows()
				
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
