#imort the packages
import numpy as np
import time
import cv2
from args_input import *
from functions import *

LABELS, weights, config, default_confidence, threshold, GPU, input_video, output_video = LineArgs()

#lista di colori per rappresentare ogni classe possibile
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype = "uint8")
#lista delle classi che vengono contate
classes = ["bicycle", "bus", "car", "motorbike", "truck"]
FRAMES_BEFORE_CURRENT = 10
inputWidth, inputHeight = 416, 416
R_col = (255, 0, 0)
G_col = (0, 255, 0)
B_col = (0, 0, 255)


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
					#count += 1
				cv2.putText(frame, str(ID), (cX, cY), cv2.FONT_HERSHEY_PLAIN, 0.5, (0,0,255), 2)
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
#fps del video sorgente
video_FPS = int(video_capture.get(cv2.CAP_PROP_FPS))
count = 0
previous_frame_detections = [{(0, 0):0} for i in range(FRAMES_BEFORE_CURRENT)]
#inizializza il video writer
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(output_video, fourcc, video_FPS, (video_width, video_height), True)
start_time = int(time.time())

#loop over frames from the video stream
while True:
	#capture frame-by-frame
	(ret, frame) = video_capture.read()
	#if the frame not grabbed, then we have reached the end
	if not ret:
		break
	#creates 4-dimensional blob from image
	#cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight), swapRB = True, crop = False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	#initialize our lists 
	IDs = []
	boxes = []
	confidences = []
	for output in layerOutputs:
		for i, detection in enumerate(output):
			scores = detection[5:]
			ID = np.argmax(scores)
			confidence = scores[ID]
			#probabiltà maggiore del minimo
			if confidence > default_confidence:
				box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
				(cX, cY, W, H) = box.astype("int")
				x = int(cX - (W/2))
				y = int(cY - (H/2))
				boxes.append([x, y, int(W), int(H)])
				confidences.append(float(confidence))
				IDs.append(ID)
	#non-maxima suppression
	nms = cv2.dnn.NMSBoxes(boxes, confidences, default_confidence, threshold)
	#disegno i rettangoli
	detectionBox(boxes, frame, IDs, confidences, nms, LABELS)
	count, detections = vehicles_count(frame, boxes, IDs, previous_frame_detections, count, nms)
	cv2.putText(frame, 'Mezzi Beccati: ' + str(count), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2, cv2.FONT_HERSHEY_COMPLEX_SMALL)

	#write the output frame to disk
	writer.write(frame)
	cv2.imshow('Frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	previous_frame_detections.pop(0)
	previous_frame_detections.append(detections)
	
print("[INFO] cleaning up...")
writer.release()
video_capture.release()
cv2.destroyAllWindows()
	

