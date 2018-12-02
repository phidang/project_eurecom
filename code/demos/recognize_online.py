import time

import cv2
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

import openface

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier


fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'openface', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

def detect_faces(img, multiple=False):
	if img is None:
		raise Exception("Unable to load image")
	rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	if multiple:
		bbs = align.getAllFaceBoundingBoxes(rgbImg)
	else:
		bb1 = align.getLargestFaceBoundingBox(rgbImg)
		bbs = [bb1]
	return bbs, rgbImg

def getRep(faces, rgbImg):
	reps = []
	for face in faces:
		alignedFace = align.align(args.imgDim, rgbImg, face,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
		if alignedFace is not None:
			rep = net.forward(alignedFace)
			reps.append((face.center().x, rep))
		else:
			raise Exception("Unable to align image")
	#sreps = sorted(reps, key=lambda x: x[0])
	return reps

def draw_bb(img, bbs, multiple=False):
	if multiple:
		for bb in bbs:
			cv2.rectangle(img, (bb.left(), bb.top()), (bb.right(), bb.bottom()), (255, 0, 255), 2)
	else:
		bb = bbs[0]
		cv2.rectangle(img, (bb.left(), bb.top()), (bb.right(), bb.bottom()), (255, 0, 255), 2)

def draw_identity(img, bbs, identity_names, confidences, multiple=False):
	font = cv2.FONT_HERSHEY_SIMPLEX
	red = (0, 0, 255)

	if (multiple):
		i=0
		for bb in bbs:
			left = (bb.left(), bb.bottom()+18)
			cv2.putText(img, identity_names[i], left, font, 0.5, red, 2) 
			up = (bb.left()+6, bb.bottom()-18)
			cv2.putText(img, str(round(confidences[i], 2)), up, font, 0.7, red, 1) 
			i=i+1
	else:
		bb = bbs[0]
		left = (bb.left()+6, bb.bottom()+18)
		cv2.putText(img, identity_names[0], left, font, 0.5, red, 2) 
		up = (bb.left(), bb.bottom()-18)
		cv2.putText(img, str(round(confidences[0], 2)), up, font, 0.7, red, 1) 

def recognize_faces(frame, clf, multiple=False):
	faces, rgbImg = detect_faces(frame['img'], multiple)
	confidences = []
	print("\n=== {} ===".format(frame['name']))

	if len(faces) > 0 and faces[0] is not None:
		draw_bb(frame['img'], faces, multiple)
		reps = getRep(faces, rgbImg)
		if len(reps) > 1:
			print("List of faces in image from left to right")
		elif len(reps) == 0:
			# print("No faces are detected.")
			return frame

		persons = []
		confidences = []
		for r in reps:
			rep = r[1].reshape(1, -1)
			bbx = r[0]
			start = time.time()
			predictions = clf.predict_proba(rep).ravel()
			maxI = np.argmax(predictions)
			person = le.inverse_transform(maxI)
			persons.append(person.decode('utf-8'))
			confidence = predictions[maxI]
			confidences.append(confidence)
			if multiple:
				print("Predict {} @ x={} with {:.2f} confidence.".format(person.decode('utf-8'),bbx,
					confidence))
			else:
				print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
			if isinstance(clf, GMM):
				dist = np.linalg.norm(rep - clf.means_[maxI])
				print("  + Distance from the mean: {}".format(dist))
		draw_identity(frame['img'], faces, persons, confidences, multiple)
	else:
		print("No faces are detected.")

	return frame, confidences

def getVideoName(directory):
	return directory.split("/")[-1].split(".")[0];	

def create_directory(output_directory, video_name, is_video_combine, threshold):
	frames_directory = output_directory + "frames_" + video_name + "/"
	video_directory = output_directory + "video_" + video_name + "/"
	threshold_directory = output_directory + "threshold_" + str(threshold) + "_" + video_name + "/"
	if not os.path.exists(frames_directory):
		os.makedirs(frames_directory)
	if not os.path.exists(video_directory) and is_video_combine:
		os.makedirs(video_directory)
	if not os.path.exists(threshold_directory):
		os.makedirs(threshold_directory)

	return frames_directory, video_directory, threshold_directory

if __name__ == '__main__':
	################# Parse Arguments ####################

	parser = argparse.ArgumentParser()

	parser.add_argument('--dlibFacePredictor', type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))

	parser.add_argument('--networkModel', type=str,
        help="Path to Torch network model.",
        default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))

	parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)

	parser.add_argument('--cuda', action='store_true')

	parser.add_argument('--videoDir', type=str, 
		help="Path to input video(s).")

	parser.add_argument('--outDir', type=str,
		help="Path to output folder.", default="./output/")

	parser.add_argument('--classifierModel', type=str,
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')

	parser.add_argument('--combineVideo',
		help='Turn on this to combine frames into video.',
        action="store_true")

	parser.add_argument('--multi', help="Infer multiple faces in image", 
		action="store_true")

	parser.add_argument('--threshold', type=float,
                        help="Threshold of probability [0-1] to save the image", default=0.9)

	args = parser.parse_args()

	#####################################################

	################# Initialization ####################
	with open(args.classifierModel, 'rb') as f:
		if sys.version_info[0] < 3:
			(le, clf) = pickle.load(f)
		else:
			(le, clf) = pickle.load(f, encoding='latin1')

	out_frames_dir, out_video_dir, out_threshold_dir = create_directory(args.outDir, 
		getVideoName(args.videoDir), args.combineVideo, args.threshold)
	
	align = openface.AlignDlib(args.dlibFacePredictor)
	net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                                  cuda=args.cuda)

	vc = cv2.VideoCapture(args.videoDir)
	cnt = 1
	rval = False

	#####################################################

	################### Face Recognition ################
	if vc.isOpened():
		rval , img = vc.read()
		if (args.combineVideo):
			height , width , layers = img.shape
			video = cv2.VideoWriter(out_video_dir + 'video.avi', 
				cv2.VideoWriter_fourcc(*'DIVX'), 24, (width,height))

	while rval:
		frame = {
	    	'img': img,
	    	'name': str(cnt)+'.jpg'
	    }
		frame, confidences = recognize_faces(frame, clf, args.multi)

		# write to frames and video
		if len(confidences)>0 and np.max(confidences)>args.threshold:
			cv2.imwrite(out_threshold_dir + frame['name'], frame['img'])
		cv2.imwrite(out_frames_dir + frame['name'], frame['img'])
		if (args.combineVideo): # Save video with prediction
			video.write(frame['img'])

		rval, img = vc.read()
	    
		cnt = cnt + 1
		cv2.waitKey(1)
	vc.release()
	
	if (args.combineVideo):
		cv2.destroyAllWindows()
		video.release()

	######################################################