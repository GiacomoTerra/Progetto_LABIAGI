import numpy as np
import argparse
import imutils
import os

def LineArgs():
	#metto su l'argparse e gli passo gli argomenti
	parser = argparse.ArgumentParser(description = "Avvia lo script")
	parser.add_argument("-i", "--input", required = True, help = "path to input video")
	parser.add_argument("-o", "--output", required = True, help = "path to output video")
	parser.add_argument("-y", "--yolo", required = True, help = "base path to YOLO directory")
	parser.add_argument("-c", "--confidence", required = False, type = float, default = 0.5, help = "minimum probability to filter weak detections")
	parser.add_argument("-t", "--threshold", required = False, type = float, default = 0.3, help = "threshold when applying non-maxima suppression")
	#argument per usare la gpu
	parser.add_argument("-g", "--gpu", required = False, type = bool, default = False)
	args = vars(parser.parse_args())
	confidence = args["confidence"]
	threshold = args["threshold"]
	GPU = args["gpu"]
	video_i = args["input"]
	video_o = args["output"]
	weights = os.path.sep.join([args["yolo"], "yolov3-320.weights"])
	config = os.path.sep.join([args["yolo"], "yolov3-320.cfg"])
	labels = os.path.sep.join([args["yolo"], "coco.names"])
	LABELS = open(labels).read().strip().split("\n")
	
	return LABELS, weights, config, confidence, threshold, GPU, video_i, video_o
