import csv
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('inputDir', type=str,
		help="Path to input csv file")

	parser.add_argument('--isLabel', type=bool,
		help="Add a title row to the label file")

	args = parser.parse_args()

	file_name = args.inputDir.split('.')[0]

	with open(args.inputDir,'r') as csvin, open(file_name + ".tsv", 'w') as tsvout:
		csvin = csv.reader(csvin)
		tsvout = csv.writer(tsvout, delimiter='\t')

		if args.isLabel:
			tsvout.writerow(['Class', 'Name'])

		for row in csvin:
			tsvout.writerow(row)