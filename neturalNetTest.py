from MLFrame.StandardImport import *
from MLFrame.NeuralNetwork import NeuralNetwok as neural

'''.......................data support........................'''
layOne = np.array([[-1,3,7],[-7,2,3],[2.4,-11,-4]])
layTwo = np.array([[6,-5.0,-1],[-1,-3,-5]])

nnet = neural.neuralNet((3,4,2))

nnet.layMatrixs[0] = layOne
nnet.layMatrixs[1] = layTwo

testx = np.array([1.1,2.2,3.3])
testvalue =  nnet(testx)

print "testvalue is %s" %testvalue
raw_input_B = raw_input("raw_input: ")

xset = np.random.rand(200,2)-0.5
xset = np.hstack((xset,np.ones(len(xset)).reshape(-1,1)))

yset = np.apply_along_axis(lambda x: nnet(x),1,xset).reshape

#yset = (xset[:,0]*2-7*xset[:,1]*xset[:,0]-15*xset[:,1]*2+2>0)

#yset = (sigmoid(3*xset[:,0]-3*xset[:,1]+0.5)>0.6)

#yset = yset*1

#plt.scatter(xset[:,0]-xset[:,1],sigmoid(yset),c='r',s=35)
#plt.show()
#print 'xset is %s' %xset

print 'yset is %s' %yset
print 'yset mean is %s' %yset.mean()

raw_input_B = raw_input("raw_input: ")

dataSet = dSet.ds_2_np(xset, yset)
nnetFc = neural.backPropagate(1000,stRate=0.01, trainData=dataSet)

trainNet = neural.neuralNet((3,3,2))

#trainNet.layMatrixs = [np.array([[1,1,1],]),]

print "trainNet is %s" %trainNet(np.array([1,2,1]))
#i=0
#it = nnetFc(trainNet)
#Dic = it(debug=False)
#funout = [Dic['hyp'](xin) for xin in dataSet.x]
#print "cost is %s"%defaultCostfun(funout, dataSet.y)
#print "parameter is %s" %Dic['hyp'].layMatrixs[0]
for Dic in nnetFc(trainNet):

	funout = np.array([Dic['hyp'](xin) for xin in dataSet.x]).flat
	#print "funout is %s"%funout
	print "cost is %s"%defaultCostfun(funout, dataSet.y)
	#raw_input("raw_input: ")
	#print "           "
		#i=0
		#print"layone is %s, lay two is %s " %(Dic['hyp'].layMatrixs[0],Dic['hyp'].layMatrixs[1])
	#print abs(Dic['hyp'](testx) - testvalue)
print"layone is %s, lay two is %s " %(trainNet.layMatrixs[0],Dic['hyp'].layMatrixs[1])
