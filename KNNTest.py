from MLFrame.StandardImport import *
from MLFrame.KNN import KNN as knn

xset = np.random.rand(200,2)-0.5
xset = np.hstack((xset,np.ones(len(xset)).reshape(-1,1)))

yset = (sigmoid(3*xset[:,0]-3*xset[:,1]+0.5)>0.6)
yset = yset*1

print "yset is %s" %yset
raw_input("input:")

dataSet = dSet.ds_2_np_patter(xset, yset)

knncls =  knn.KNNClassify(10,dataSet)


xsettest = np.random.rand(100,2)-0.5
xsettest = np.hstack((xsettest,np.ones(len(xsettest)).reshape(-1,1)))
ysettest = (sigmoid(3*xsettest[:,0]-3*xsettest[:,1]+0.5)>0.6)
ysettest = ysettest*1

differ = (np.array([ knncls(xin) for xin in  xsettest])-ysettest).sum()/len(yset)

print "cost is %s" %differ