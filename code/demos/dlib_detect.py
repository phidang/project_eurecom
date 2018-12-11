import time

import cv2
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

import openface
import time

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier


fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'openface', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

total_time = 0.0
min_time = 100.0
max_time = 0.0
total_faces = 0

def get_bounding_box(img, multi_faces=False):
	global total_time, min_time, max_time
	if img is None:
		raise Exception("Unable to load image")
	rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	if multi_faces:
		start = time.time()
		bbs = align.getAllFaceBoundingBoxes(rgbImg)
		ctime = time.time()-start
		total_time = total_time + ctime
		min_time = min(min_time, ctime)
		max_time = max(max_time, ctime)
	else:
		bb1 = align.getLargestFaceBoundingBox(rgbImg)
		bbs = [bb1]
	return bbs

def draw_bb(img, bbs, multi_faces=False):
	if multi_faces:
		for bb in bbs:
			cv2.rectangle(img, (bb.left(), bb.top()), (bb.right(), bb.bottom()), (255, 0, 255), 2)
	else:
		bb = bbs[0]
		cv2.rectangle(img, (bb.left(), bb.top()), (bb.right(), bb.bottom()), (255, 0, 255), 2)

def detect_faces(frame, multi_faces=False):
	global total_faces

	faces = get_bounding_box(frame['img'], multi_faces)
	print("\n=== {} ===".format(frame['name']))

	if len(faces) > 0 and faces[0] is not None:
		draw_bb(frame['img'], faces, multi_faces)
		total_faces = total_faces + len(faces)
	else:
		print("No faces are detected.")
		return -1

	return frame

def getVideoName(directory):
	return directory.split("/")[-1].split(".")[0];	

def create_directory(output_directory, video_name):
	frames_directory = output_directory + "frames_" + video_name + "/"
	if not os.path.exists(frames_directory):
		os.makedirs(frames_directory)
	return frames_directory

if __name__ == '__main__':
	global total_frames
	total_frames = 0

	################# Parse Arguments ####################

	parser = argparse.ArgumentParser()

	parser.add_argument('--dlibFacePredictor', type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))

	parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)

	parser.add_argument('--videoDir', type=str, 
		help="Path to input video(s).")

	parser.add_argument('--outDir', type=str,
		help="Path to output folder.", default="./output/")

	parser.add_argument('--multi_faces', help="Infer multiple faces in image", 
		action="store_true")

	args = parser.parse_args()

	#####################################################

	################# Initialization ####################

	out_frames_dir = create_directory(args.outDir, 
		getVideoName(args.videoDir))
	
	align = openface.AlignDlib(args.dlibFacePredictor)

	vc = cv2.VideoCapture(args.videoDir)
	cnt = 1
	rval = False

	#####################################################

	################### Face Recognition ################
	if vc.isOpened():
		rval , img = vc.read()

	while rval:
		frame = {
	    	'img': img,
	    	'name': str(cnt)+'.jpg'
	    }
		frame = detect_faces(frame, args.multi_faces)
		total_frames = total_frames + 1

		# write to frames
		if frame!=-1:
			cv2.imwrite(out_frames_dir + frame['name'], frame['img'])

		rval, img = vc.read()
	    
		cnt = cnt + 1
		cv2.waitKey(1)
	vc.release()
	
	print("Video:", getVideoName(args.videoDir))
	print("Total time:", total_time)
	print("Minimum time:", min_time)
	print("Maximum time:", max_time)
	print("Total frames:", total_frames)
	print("Total detected faces:", total_faces)
	######################################################