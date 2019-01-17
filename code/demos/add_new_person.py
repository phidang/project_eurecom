import argparse
import os
import shutil
import subprocess

def clear_alignment_directory(directory):
	for the_file in os.listdir(directory):
		file_path = os.path.join(directory, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path): shutil.rmtree(file_path)
		except Exception as e:
			print(e)

def merge_reps_files(old_dir, new_dir, file_name):
	fout = open(old_dir + file_name, "a")
	f = open(new_dir + file_name)
	f.next() # skip the header
	for line in f:
		fout.write(line)
	f.close() # not really needed

def merge_label_files(old_dir, new_dir, file_name):
	line = subprocess.check_output(['tail', '-1', old_dir + file_name])
	new_label = int(line.split(',')[0]) + 1
	fout = open(old_dir + file_name, "a")
	f = open(new_dir + file_name)
	f.next() # skip the header
	for line in f:
		fout.write(str(new_label) + ',' + line.split(',')[1])
	f.close() # not really needed

if __name__ == '__main__':
	################# Parse Arguments ####################

	parser = argparse.ArgumentParser()

	parser.add_argument('--inputDir', type=str, 
		help='Path to new person folder. Has to be placed in directory "./data/new_person/". Please remove the previous persons from this folder.',
		default="./data/new_person/")

	parser.add_argument('--alignDir', type=str, 
		help="Path to the output aligned folder, has to be different from the aligned folder of embedding.",
		default="./data/new_person_align/")

	parser.add_argument('--featureDir', type=str,
		help="Path to the feature folder that the new person will be added.",
		default="./features/")

	args = parser.parse_args()

	align_command = 'python ./demos/align-dlib.py ' + args.inputDir + ' align outerEyesAndNose ' + args.alignDir + ' --size 96'

	prune_command = 'python openface/util/prune-dataset.py ' + args.alignDir + ' --numImagesThreshold 1'

	embed_command = './openface/batch-represent/main.lua -outDir ' + args.alignDir + ' -data ' + args.alignDir 

	train_command = 'python ./demos/train.py train --classifier KNN ' + args.featureDir

	# print(align_command)
	# print(prune_command)
	# print(embed_command)
	# print(train_command)

	os.system(align_command)
	print("-------- Alignment Completed ----------")
	os.system(prune_command)
	print("-------- Pruning Completed ----------")
	os.system(embed_command)
	print("-------- Embedding Generating Completed ----------")

	merge_reps_files(args.featureDir, args.alignDir, "reps.csv")
	merge_label_files(args.featureDir, args.alignDir, "labels.csv")
	clear_alignment_directory(args.alignDir)

	os.system(train_command) # Re-train the classifier
	print("-------- Classifier Training Completed ----------")



