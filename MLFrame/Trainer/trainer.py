#Training mod

def  squareDValue(a,b):
	return ((a - b)**2).sum()/len(a)

class trainer(object):
	def defaultCostFun(self,x,yout):
			return squareDValue(trainset,yout)

	def __init__(self,trainSet,costFunc=defaultCostFun):
		super.__init__()
		self.costFunc = costFunc
		self.trainSet = trainSet

	def excTraining(self,trainset,hypothesis,**trainpara):
		yield self.mCostFunction(trainset,hypothesis(trainset[:,0])),hypothesis
		
	def __call__(self,hypothesis):
		return self.excTraining(self.trainset, hypothesis)
	