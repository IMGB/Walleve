from ..StandardImport import *

class neuralNet(hyp.hypothesis):
	"""docstring for neuralNet"""

	def __init__(self,lays):
		super(neuralNet, self).__init__()
		self._mve = False
		self.setShape(lays)

	@property
	def shape(self):
		retlen = [self.layMatrixs[0].shape[1],]#bias unit
		[retlen.append(len(arr)) for arr in  self.layMatrixs]  
		return retlen
	
	def setShape(self, lays):
		if len(lays) < 2:
			raise TypeError("imquire lay number more than 3")
		else:
			self.layMatrixs = []
		for i in range(1,len(lays)):
			self.layMatrixs.append(np.random.rand(lays[i],lays[i-1])-0.5)
			#self.layMatrixs.append(np.zeros((lays[i],lays[i-1])))
		return

	@property
	def inputNum(self):
		return self.layMatrixs[0].shape[1]

	@property
	def outputNum(self):
		return self.layMatrixs[-1].shape[0]

	def excution(self,xin,retAllUnit = False):
		retUnits = [xin,] 
		unit_in = xin
		for i in range(len(self.layMatrixs)):
			unit_in = sigmoid((self.layMatrixs[i]*unit_in).sum(axis = self.layMatrixs[i].ndim-1))
			retUnits.append(unit_in)

		if retAllUnit:	return retUnits
		else:		return unit_in
	
	def checkXin(self,xin):
		return xin.size == self.inputNum

class backPropagate(train.factory):
	"""docstring for backpropagate"""
	def __init__(self,maxStep,stRate=0.01,trainData=None,checkData=None):
		super(backPropagate, self).__init__(trainData=trainData, checkData=checkData)
		self.stRate = stRate
		self.maxStep = maxStep
		
	def excFactory(self,it):
		it.trainMaxStep = self.maxStep
		it.stRate = self.stRate
		dslen = len(it.trainData)*1.0
		layMatrixs = it.hyp.layMatrixs
		def excTraining(it):
			#deltParams = [np.zeros_like(matrix) for matrix in layMatrixs]
			for xin,yin in it.trainData:
				aUnits = it.hyp.excution(xin,retAllUnit = True)
				layNum = len(it.hyp.layMatrixs)
				delta = yin - aUnits[-1]
				for i in range(layNum-1,-1,-1):
					aUnit = aUnits[i]
					aUint_delt = aUnits[i+1]*(1-aUnits[i+1])
					delta = delta*aUint_delt
					detPrars = delta.reshape(-1,1)*aUnit
					#print "lay is %d delta is %s detPrars is %s" %(i,delta,detPrars)
					#new delta
					delta = np.dot(delta, layMatrixs[i])
					deltaMatrix = detPrars*it.stRate
					layMatrixs[i] =layMatrixs[i]  + deltaMatrix

			'''
			for i in range(len(deltParams)):
				average_deltParas = deltParams[i]/dslen
				layMatrixs[i] =layMatrixs[i] + (average_deltParas*it.stRate)

				#print "average_deltParas is %s" %average_deltParas
				#print "delta is %s" %(average_delt/dslen)
				#print "layMatrixs[i] is %s" %layMatrixs[i]
			'''

		return excTraining
	