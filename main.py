#imort the packages
import numpy as np
import time
import cv2
from args_input import *
from function import *

LABELS, weights, config, confidence, threshold, GPU, input_video, output_video = LineArgs()

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

#inizializza lo stream video e prende le dimensioni del frame
video_capture = cv2.VideoCapture(input_video)
video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#fps del video sorgente
video_FPS = int(video_capture.get(cv2.CAP_PROP_FPS))

if writer is None
	#inizializza il video writer
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	writer = cv2.VideoWriter(output_video, forcc, video_FPS, (video_width, video_height), True)


