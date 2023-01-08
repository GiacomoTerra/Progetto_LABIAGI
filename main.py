#imort the packages
import numpy as np
import argparse
import time
import cv2
import os
import imutils
from functions import *
from tracker import *
from tqdm import tqdm
from collections import Counter

#metto su l'argparse e gli passo gli argomenti
parser = argparse.ArgumentParser(description = "Avvia lo script")
parser.add_argument("-i", "--input", required = True, help = "path to input video")
parser.add_argument("-o", "--output", required = True, help = "path to output video")
parser.add_argument("-y", "--yolo", required = True, help = "base path to YOLO directory")
parser.add_argument("-il", "--inlog", default = "logs", help = "base  path to IN log file")
parser.add_argument("-ol", "--outlog", default = "logs", help = "base path to OUT log file")
parser.add_argument("-c", "--confidence", required = False, type = float, default = 0.7, help = "minimum probability to filter weak detections")
parser.add_argument("-t", "--threshold", required = False, type = float, default = 0.3, help = "threshold when applying non-maxima suppression")
#argument per usare la gpu
parser.add_argument("-g", "--gpu", required = False, type = bool, default = False)
args = vars(parser.parse_args())
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
weights = os.path.sep.join([args["yolo"], "yolov3-320.weights"])
config = os.path.sep.join([args["yolo"], "yolov3-320.cfg"])
input_video = args["input"]
GPU = args["gpu"]
default_confidence = args["confidence"]
threshold = args["threshold"]

#funzione per pulire temp logs
def clear_logs():
	if os.path.exists(os.path.sep.join([args["inlog"], "In.txt"])):
		os.remove(os.path.sep.join([args["inlog"], "In.txt"]))
	if os.path.exists(os.path.sep.join([args["outlog"], "Out.txt"])):
		os.remove(os.path.sep.join([args["outlog"], "Out.txt"]))

#inizializzo il tracker
tracker = Tracker()
#lista di colori per rappresentare ogni classe possibile
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype = "uint8")
#lista delle classi che vengono contate
classes = ["bicycle", "bus", "car", "motorbike", "truck"]
FRAMES_BEFORE_CURRENT = 10
inputWidth, inputHeight = 416, 416
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

#funzione che conta il numero totale dei veicoli
def vehicles_count(frame, boxes, IDs, frame_pre, count, i):
	detections = {}
	if len(i) > 0:
		for k in i.flatten():
			(x, y) = (boxes[k][0], boxes[k][1])
			(w, h) = (boxes[k][2], boxes[k][3])
			cX = x + (w//2)
			cY = y + (h//2)
			if LABELS[IDs[k]] in classes:
				detections[(cX, cY)] = count
				if not BoxGiaPresente((cX, cY, w, h), detections, frame_pre):
					count += 1
				ID = detections.get((cX, cY))
				if list(detections.values()).count(ID) > 1:
					detections[(cX, cY)] = count
					count += 1
				cv2.putText(frame, str(ID), (cX, cY), cv2.FONT_HERSHEY_PLAIN, 0.5, BLUE, 2)
	return count, detections
	
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(config, weights)

#using GPU
if GPU:
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#determine only the output layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
#inizializza lo stream video e prende le dimensioni del frame
video_capture = cv2.VideoCapture(input_video)
video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_FPS = int(video_capture.get(cv2.CAP_PROP_FPS))
(W, H) = (None, None)
writer = None
count = 0
previous_frame_detections = [{(0, 0):0} for i in range(FRAMES_BEFORE_CURRENT)]

#prova a determinare il numero totale di frame 
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(video_capture.get(prop))
	print("[INFO] Total frames in the video: {}".format(total))
except:
	print("[INFO] Could not determinate num of frames")
	total = -1
clear_logs()

start_time = int(time.time())

#loop over frames from the video stream
for fr in tqdm(range(total)):   
	while True:
		#capture frame-by-frame
		(ret, frame) = video_capture.read()
		#if the frame not grabbed, then we have reached the end
		if not ret:
			break
		
		if W is None or H is None:
			(H, W) = frame.shape[:2]
			
		#creates 4-dimensional blob from image
		#cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight), swapRB = True, crop = False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()
		#initialize our lists 
		freq = []
		IDs = []
		boxes = []
		confidences = []
		var = []
		for output in layerOutputs:
			for detection in output:
				scores = detection[5:]
				ID = np.argmax(scores)
				confidence = scores[ID]
				#probabiltà maggiore del minimo
				if confidence > default_confidence:
					box = detection[0:4] * np.array([W, H, W, H])
					(cX, cY, width, height) = box.astype("int")
					x = int(cX - (width/2))
					y = int(cY - (height/2))
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					IDs.append(ID)
		#non-maxima suppression
		nms = cv2.dnn.NMSBoxes(boxes, confidences, default_confidence, threshold)
		
		#disegno i rettangoli di identificazione
		detectionBox(boxes, frame, IDs, confidences, nms, LABELS)
		count, detections = vehicles_count(frame, boxes, IDs, previous_frame_detections, count, nms)		
		cv2.putText(frame, 'Veicoli Individuati: ' + str(count), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.9, GREEN, 2, cv2.FONT_HERSHEY_COMPLEX_SMALL)	
		
		if len(nms) > 0:
			var = []
			var1 = 0
			for i in nms.flatten():
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				freq1 = [j for j in IDs]
				freq = dict([LABELS[x], IDs.count(x)] for x in set(IDs))
				freq = str(freq)[1:-1]
				text1 = ("Overall Vehicles in Frame = {}".format(freq))
				cv2.putText(frame, text1, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 2)
				cv2.putText(frame, "IN: ",(363, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLUE, 2)
				cv2.putText(frame, "OUT: ", (850, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLUE, 2)
				cv2.line(frame, (415, 361), (615, 361), (0, 0, 0xFF), 3)
				cv2.line(frame, (670, 361), (843, 361), (0, 0, 0xFF), 3)
				var = 1
				#Detected vehicles in Lane 1
				if (x > 415 and x < 620 and y > 360 and y < 362.5):
					cv2.line(frame, (415, 361), (615, 361), (0, 0xFF, 0), 7)
					f = open(os.path.sep.join([args["inlog"], "In.txt"]), "a")
					f.write("%d\n" %var)
					f.close()
					var1 = sum([int(s.strip()) for s in open(os.path.sep.join([args["inlog"], "In.txt"]), "r").readlines()])
					text2 = "{}".format(var1)
					cv2.putText(frame, text2, (390, 355), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0xFF), 2)
					f.close()
				else:
					try:
						var1 = sum([int(s.strip()) for s in open(os.path.sep.join([args["inlog"], "In.txt"]), "r").readlines()])
						text2 = "{}".format(var1)
						cv2.putText(frame, text2, (390, 355), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0xFF), 2)
						f.close()
					except:
						var1 = 0
						text2 = "{}".format(var1)
						cv2.putText(frame, text2, (390, 355), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0xFF), 2)
				#Detected vehicles in Lane 2
				if (x > 670 and x < 845 and y > 360 and y < 362.5):
					cv2.line(frame, (670, 361), (843, 361), (0, 0xFF, 0), 7)
					f = open(os.path.sep.join([args["outlog"], "Out.txt"]), "a")
					f.write("%d\n" %var)
					f.close()
					var1 = sum([int(s.strip()) for s in open(os.path.sep.join([args["outlog"], "Out.txt"]), "r").readlines()])
					text2 = "{}".format(var1)
					cv2.putText(frame, text2, (895, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0xFF), 2)
					f.close()
				else:
					try:
						var1 = sum([int(s.strip()) for s in open(os.path.sep.join([args["outlog"], "Out.txt"]), "r").readlines()])
						text2 = "{}".format(var1)
						cv2.putText(frame, text2, (895, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0xFF), 2)
						f.close()
					except:
						var1 = 0
						text2 = "{}".format(var1)
						cv2.putText(frame, text2, (895, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0xFF), 2)
			
		
		#controllo se il writer is None
		if writer is None:
			#inizializzo il writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)
			if total > 0:
				elap = end - start
				print("[INFO] Estimated time taken to process single frame: {:.4f}seconds".format(elap))
				print("\n[INFO] Estimated total time to finish object detection: {:.4f}minutes".format((elap * total)/60))
				print("\n Object Detection Progress: ")      
		#write the output frame to disk
		writer.write(frame)
		cv2.imshow('Frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		previous_frame_detections.pop(0)
		previous_frame_detections.append(detections)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
clear_logs()
	
print("\n[INFO] cleaning up...")
print("\n[INFO] Detection completed")
writer.release()
video_capture.release()
cv2.destroyAllWindows()
	

