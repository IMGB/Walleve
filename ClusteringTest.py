from MLFrame.StandardImport import *
from MLFrame.Clustering import clustering as cluster
from scipy.stats import norm

'''      ...................ready data'''
point1 = norm.rvs(loc = 1,size = (50,2))
point2 = norm.rvs(loc = 5,size = (50,2))
point3 = norm.rvs(loc = 9,size = (50,2))

#scatter(x, y, s=size, c=colors)
xset = np.concatenate((point1,point2,point3),axis=0)
plt.scatter(xset[:,0],xset[:,1])
points = np.array([[2.0,7.0],[4.0,1.0],[9.0,5.0]])


myCluster = cluster.clusterClassify(points)
myTrain = cluster.clusteringTrain(xset)
it = myTrain(myCluster)
for i in range(100):
	print  it.next()['hyp'].clusters

plt.scatter(it.hyp.clusters[:,0],it.hyp.clusters[:,1],c='r',s=35)
plt.show()


pointTest = norm.rvs(loc = 9,size = 2)



print "Result %d" %myCluster(pointTest)