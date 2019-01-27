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

def merge_label_files_1(old_dir, new_dir, file_name):
	line = subprocess.check_output(['tail', '-1', old_dir + file_name])
	base_label_num = int(line.split(',')[0])
	fout = open(old_dir + file_name, "a")
	f = open(new_dir + file_name)
	f.next() # skip the header
	for line in f:
		split_line = line.split(',')
		new_label = base_label_num + int(split_line[0])
		fout.write(str(new_label) + ',' + split_line[1])
	f.close() # not really needed

# function to create a map (label, person_name)
def create_map(old_dir, file_name):
	fout = open(old_dir + file_name, "r")
	f_map = open(old_dir + "maps.csv", "w")
	labels = fout.readlines()
	prev_label = ""
	for line in labels:
		curr_label = line.split(',')[0] # get the current label
		if curr_label != prev_label:
			person_name = line.split(',')[1].split('/')[-2] # get the person name
			f_map.write(curr_label + ',' + person_name + '\n')
			prev_label = curr_label
	fout.close()
	f_map.close()

def merge_label_files(old_dir, new_dir, file_name):
	# check if there is already a map or not
	if not os.path.isfile(old_dir + "maps.csv"):
		create_map(old_dir, file_name)

	# get the map from file
	f_map = open(old_dir + "maps.csv", "r")
	maps = f_map.readlines()
	f_map.close()

	fout = open(old_dir + file_name, "a")
	f = open(new_dir + file_name)
	f.next() # skip the header
	prev_label = ""
	save_label = ""
	for line in f:
		split_line = line.split(',')
		if (split_line[0] == prev_label):
			fout.write(str(save_label) + ',' + split_line[1])
			continue
		person_name = split_line[1].split('/')[-2]
		label = [s for s in maps if person_name in s]
		if not label: # this is new person
			label = int(maps[-1].split(',')[0]) + 1
			maps.append(str(label) + ',' + person_name + '\n')
		else:
			label = label[0].split(',')[0]
		fout.write(str(label) + ',' + split_line[1])
		save_label = label
	f.close() # not really needed

	# write back the map to file
	f_map = open(old_dir + "maps.csv", "w")
	f_map.writelines(maps)
	f_map.close()

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

	parser.add_argument('--classifierModel', type=str,
		help="Define the classifier to be re-trained.",
		default="KNN")

	args = parser.parse_args()

	align_command = 'python ./demos/align-dlib.py ' + args.inputDir + ' align outerEyesAndNose ' + args.alignDir + ' --size 96'

	prune_command = 'python openface/util/prune-dataset.py ' + args.alignDir + ' --numImagesThreshold 1'

	embed_command = './openface/batch-represent/main.lua -outDir ' + args.alignDir + ' -data ' + args.alignDir 

	train_command = 'python ./demos/train.py train --classifier ' + args.classifierModel + ' ' + args.featureDir

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



