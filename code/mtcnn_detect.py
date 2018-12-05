from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import cv2
import argparse
import os
import pickle
import sys

import numpy as np
import tensorflow as tf
import facenet
import align.detect_face
import time
import random

from time import sleep


fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'openface', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

def draw_bb(img, bbs):
	for bb in bbs:

		cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255, 0, 255), 2)

def getVideoName(directory):
	return directory.split("/")[-1].split(".")[0];	

def create_directory(output_directory, video_name):
	frames_directory = output_directory + "frames_" + video_name + "/"
	if not os.path.exists(frames_directory):
		os.makedirs(frames_directory)
	return frames_directory

if __name__ == '__main__':
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

	args = parser.parse_args()

	#####################################################

	################# Initialization ####################

	out_frames_dir = create_directory(args.outDir, 
		getVideoName(args.videoDir))
	
	with tf.Graph().as_default():
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		with sess.as_default():
			pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

	minsize = 20 # minimum size of face
	threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
	factor = 0.709 # scale factor

	vc = cv2.VideoCapture(args.videoDir)
	cnt = 1
	rval = False

	total_frames = 0
	total_faces = 0
	total_time = 0.0
	min_time = 100.0
	max_time = 0.0
	#####################################################

	################### Face Recognition ################
	if vc.isOpened():
		rval , img = vc.read()

	while rval:
		frame = {
	    	'img': img,
	    	'name': str(cnt)+'.jpg'
	    }

		rgbImg = cv2.cvtColor(frame['img'], cv2.COLOR_BGR2RGB)
		start = time.time()

		faces, _ = align.detect_face.detect_face(frame['img'], minsize, pnet, rnet, onet, threshold, factor)
		
		ctime = time.time()-start
		total_time = total_time + ctime
		min_time = min(min_time, ctime)
		max_time = max(max_time, ctime)

		print("\n=== {} ===".format(frame['name']))

		if len(faces) > 0 and faces[0] is not None:
			draw_bb(frame['img'], faces)
			total_faces = total_faces + faces.shape[0]
		else:
			print("No faces are detected.")

		total_frames = total_frames + 1

		# write to frames
		if faces.shape[0]>0:
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