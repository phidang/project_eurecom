import matplotlib.pyplot as plt

f = open("./demos/plot.txt", "r") 

x = []
y = []
labels = []
n = int(f.readline())
m = int(f.readline())
for i in range(n):
	s = f.readline().split(' ')
	x.append(s[0])
	y.append(s[1])
	labels.append(s[2])

plt.figure(figsize=(16, 16)) 
colors = ['red', 'green', 'blue', 'black']
for i in range(len(x)):
	plt.scatter(x[i], y[i], c=colors[int(labels[i])])
	plt.annotate(labels[i],
		xy=(x[i], y[i]),
		xytext=(10, -20),
		textcoords='offset points',
		ha='right',
		va='bottom')
	
plt.show()