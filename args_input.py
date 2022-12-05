import numpy as np
import argparse
import imutils
import os

def LineArgs():
	#metto su l'argparse e gli passo gli argomenti
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", required = True)
	parser.add_argument("-o", "--output", required = True)
	parser.add_argument("-y", "--yolo", required = True)
	parser.add_argument("-c", "--confidence", required = False, type = float, default = 0.5)
	parser.add_argument("-t", "--threshold", required = False, type = float, default = 0.3)
	#argument per usare la gpu
	parser.add_argument("-g", "--gpu", required = False, type = bool, default = False)
	args = vars(parser.parse_args())
	confidence = args["confidence"]
	threshold = args["threshold"]
	gpu = args["gpu"]
	video_i = args["input"]
	video_o = args["output"]
	weights = os.path.sep.join([args["yolo"], "yolov3-320.weights"])
	config = os.path.sep.join([args["yolo"], "yolov3-320.cfg"])
	labels = os.path.sep.join([args["yolo"], "coco.names"])
	Labels = open(labels).read().strip().split("\n")
	
	return Labels, weights, config, confidence, threshold, gpu
