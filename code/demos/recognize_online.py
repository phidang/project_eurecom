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
from sklearn.neighbors import KNeighborsClassifier


fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'openface', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

def crop_face_from_image(img, face):
	top = max(0, face.top()-10)
	bottom = min(face.bottom()+10, img.shape[0]-1)
	left = max(0, face.left()-10)
	right = min(face.right()+10, img.shape[1]-1)
	crop_face = img[top:face.bottom()+10, left:face.right()+10]
	return crop_face

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
	aligned_faces = []
	for face in faces:
		alignedFace = align.align(args.imgDim, rgbImg, face,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
		aligned_faces.append(alignedFace)
		if alignedFace is not None:
			rep = net.forward(alignedFace)
			reps.append((face.center().x, rep))
		else:
			raise Exception("Unable to align image")
	#sreps = sorted(reps, key=lambda x: x[0])
	return reps, aligned_faces

def draw_bb(img, bbs, multiple=False):
	green = (0, 255, 0)
	if multiple:
		for bb in bbs:
			cv2.rectangle(img, (bb.left(), bb.top()), (bb.right(), bb.bottom()), green, 2)
	else:
		bb = bbs[0]
		cv2.rectangle(img, (bb.left(), bb.top()), (bb.right(), bb.bottom()), green, 2)

def draw_identity(img, bbs, identity_names, confidences, recognizeFace, multiple=False):
	font = cv2.FONT_HERSHEY_SIMPLEX
	red = (0, 0, 255)

	if (multiple):
		i=0
		for bb in bbs:
			if recognizeFace == "" or identity_names[i]==recognizeFace:
				left = (bb.left(), bb.bottom()+18)
				cv2.putText(img, identity_names[i], left, font, 0.5, red, 2) 
				up = (bb.left()+6, bb.bottom()-18)
				#cv2.putText(img, str(round(confidences[i], 2)), up, font, 0.7, red, 1) 
				if identity_names[i]==recognizeFace: # draw another bounding box to show the difference
					cv2.rectangle(img, (bb.left(), bb.top()), (bb.right(), bb.bottom()), red, 2)
			i=i+1
	else:
		bb = bbs[0]
		left = (bb.left()+6, bb.bottom()+18)
		cv2.putText(img, identity_names[0], left, font, 0.5, red, 2) 
		up = (bb.left(), bb.bottom()-18)
		#cv2.putText(img, str(round(confidences[0], 2)), up, font, 0.7, red, 1) 

def recognize_faces(frame, clf, recognizeFace, multiple=False):
	faces, rgbImg = detect_faces(frame['img'], multiple)
	confidences = []
	persons = []
	print("\n=== {} ===".format(frame['name']))

	if len(faces) > 0 and faces[0] is not None:
		draw_bb(frame['img'], faces, multiple)
		reps, align_faces = getRep(faces, rgbImg)
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
			if isinstance(clf, GMM):
				dist = np.linalg.norm(rep - clf.means_[maxI])
				print("  + Distance from the mean: {}".format(dist))
			if isinstance(clf, KNeighborsClassifier):
				dist, ind = clf.kneighbors(rep, 1)
				confidence = dist[0][0]
				confidences.append(confidence)
				print("Predict {} @ x={} with {:.2f} confidence.".format(person.decode('utf-8'), bbx, confidence))
			else:
				confidence = predictions[maxI]
				confidences.append(confidence)
				if multiple:
					print("Predict {} @ x={} with {:.2f} confidence.".format(person.decode('utf-8'),bbx,
						confidence))
				else:
					print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))

		draw_identity(frame['img'], faces, persons, confidences, recognizeFace, multiple)
	else:
		print("No faces are detected.")

	return frame, confidences, faces, persons

def getVideoName(directory):
	return directory.split("/")[-1].split(".")[0];	

def create_directory(args):
	video_name = getVideoName(args.videoDir)
	frames_directory = args.outDir + "frames_" + video_name + "/"
	video_directory = args.outDir + "video_" + video_name + "/"
	threshold_directory = args.outDir + "threshold_" + str(args.threshold) + "_" + video_name + "/"
	faces_directory = args.outDir + "faces_" + video_name + "/" 
	if not os.path.exists(frames_directory) and args.saveAllFrames:
		os.makedirs(frames_directory)
	if not os.path.exists(video_directory) and args.combineVideo:
		os.makedirs(video_directory)
	if not os.path.exists(faces_directory) and args.saveFaces:
		os.makedirs(faces_directory)
	if not os.path.exists(threshold_directory) and args.threshold>=0.0:
		os.makedirs(threshold_directory)

	return frames_directory, video_directory, threshold_directory, faces_directory

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

	parser.add_argument('--saveAllFrames',
		help='Turn on this to save all frames',
        action="store_true")

	parser.add_argument('--saveFaces',
		help='Turn on this to save faces with their corresponding predictions',
        action="store_true")

	parser.add_argument('--combineVideo',
		help='Turn on this to combine frames into video.', 
		action="store_true")

	parser.add_argument('--multi', help="Infer multiple faces in image", 
		action="store_true")

	parser.add_argument('--threshold', type=float,
		help="Threshold of probability [0-1] to save the image", default=-1.0)

	parser.add_argument('--resizeVideoRatio', type=float,
		help="Resize input video by a ratio. A float number required.", default=1.0)

	parser.add_argument('--recognizeFace', type=str,
		help="Name of the person needs to be recognized.", default="")

	args = parser.parse_args()

	#####################################################

	################# Initialization ####################
	with open(args.classifierModel, 'rb') as f:
		if sys.version_info[0] < 3:
			(le, clf) = pickle.load(f)
		else:
			(le, clf) = pickle.load(f, encoding='latin1')

	out_frames_dir, out_video_dir, out_threshold_dir, out_faces_dir = create_directory(args)
	
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
		img = cv2.resize(img, (0, 0), fx=args.resizeVideoRatio, fy=args.resizeVideoRatio)
		if (args.combineVideo):
			height , width , layers = img.shape
			video = cv2.VideoWriter(out_video_dir + 'video.avi', 
				cv2.VideoWriter_fourcc(*'DIVX'), 24, (width,height))

	while rval:
		# Resize each frame in the video by a ratio
		img = cv2.resize(img, (0, 0), fx=args.resizeVideoRatio, fy=args.resizeVideoRatio)
		frame = {
	    	'img': img.copy(),
	    	'name': str(cnt)+'.jpg'
	    }
		frame, confidences, faces, persons = recognize_faces(frame, clf, args.recognizeFace, args.multi)

		# write to frames and video
		if args.threshold>=0.0 and len(confidences)>0 and np.max(confidences)>args.threshold:
		# and "BrigitteBardot" in persons:
			cv2.imwrite(out_threshold_dir + frame['name'], frame['img'])

		# crop face and save it with its corresponding person's name
		if args.saveFaces and len(faces)>0:
			i = 0
			for face in faces:
				crop_face = crop_face_from_image(img, face)
				#save_dir = out_faces_dir + persons[i] + "/"
				crop_save_dir = out_faces_dir + persons[i] + "_" + str(format(confidences[i],'.4f')) + "_" + str(cnt) + ".jpg"
				cv2.imwrite(crop_save_dir, crop_face)
				# cv2.imwrite(save_dir_sorted + str(round(confidences[i],2)) + "_" + str(cnt) + ".jpg", crop_face)
				i = i+1

		if args.saveAllFrames:
			cv2.imwrite(out_frames_dir + frame['name'], frame['img'])

		if args.combineVideo: # Save video with prediction
			video.write(frame['img'])

		rval, img = vc.read()
	    
		cnt = cnt + 1
		cv2.waitKey(1)
	vc.release()
	
	if (args.combineVideo):
		cv2.destroyAllWindows()
		video.release()

	######################################################