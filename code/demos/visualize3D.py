from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

f = open("./demos/plot.txt", "r") 

x = []
y = []
z = []
labels = []
n = int(f.readline())
m = int(f.readline())
for i in range(n):
	s = f.readline().split(' ')
	x.append(float(s[0]))
	y.append(float(s[1]))
	if m==3:
		z.append(float(s[2]))
		labels.append(s[3])
	else:
		labels.append(s[2])

colors = ['red', 'green', 'blue', 'yellow']
for i in range(4):
	print(i, colors[i])

fig = plt.figure(figsize=(16, 16)) 
if m==3:
	ax = fig.add_subplot(111, projection='3d')
for i in range(len(x)):
	if m == 3:
		ax.scatter(x[i], y[i], z[i], c=colors[int(labels[i])])
	else:
		plt.scatter(x[i], y[i], c=colors[int(labels[i])])

plt.show()
	# ax.annotate(labels[i],
	# 	xy=(x[i], y[i]),
	# 	xytext=(10, -20),
	# 	textcoords='offset points',
	# 	ha='right',
	# 	va='bottom')
	
