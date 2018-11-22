import os
import pandas as pd
import argparse

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

from operator import itemgetter

def get_Embedding(args):
	print("Loading embeddings.")
	fname = "{}/labels.csv".format(args.workDir)
	labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
	print(labels[1])
	labels = map(itemgetter(1),
		map(os.path.split,
		map(os.path.dirname, labels)))  # Get the directory.
	fname = "{}/reps.csv".format(args.workDir)
	embeddings = pd.read_csv(fname, header=None).as_matrix()
	le = LabelEncoder().fit(labels)
	labelsNum = le.transform(labels)
	nClasses = len(le.classes_)
	print("Embeddings have {} classes.".format(nClasses))
	for i in range(nClasses):
		print(list(le.classes_)[i], i)

	return embeddings, labelsNum

def tsne_plot(embeddings, labels, dim, perplexity):
	tsne_model = TSNE(perplexity=perplexity, n_components=dim, init='pca', n_iter=500, random_state=10)
	new_values = tsne_model.fit_transform(embeddings)
	
	# Since we are using python2.7 to use OpenFace Library,
	# but the matplotlib should be called in python3, 
	# so we need to save the data for plotting to file 
	# and call the visualize script by python 3
	f = open("./demos/plot.txt","w")
	f.write(str(len(labels))+'\n')
	f.write(str(dim)+'\n')
	i = 0
	for value in new_values:
		f.write(str(value[0])+' ')
		f.write(str(value[1])+' ')
		if dim==3:
			f.write(str(value[2])+' ')
		f.write(str(labels[i])+'\n')
		i=i+1
	f.close()

	if dim==3: 
		os.system("python3 ./demos/visualize3D.py")
	else:
		os.system("python3 ./demos/visualize.py")

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('workDir', type=str,
		help="The input work directory containing 'reps.csv' and 'labels.csv'. Obtained from aligning a directory with 'align-dlib' and getting the representations with 'batch-represent'.")

	parser.add_argument('--dim', type=int,
		help="Number of components that need to be visualized.",
		default=2)

	parser.add_argument('--perplex', type=int,
		help="Perplexity of TSNE, usually in range [5-50], larger for larger dataset.",
		default=10)

	args = parser.parse_args()

	embeddings, labels = get_Embedding(args)

	tsne_plot(embeddings, labels, args.dim, args.perplex)