from ..StandardImport import *

class LinearHyp(hyp.hypothesis):
	""""""
	def __init__(self,featureNum=1,params=None):
		if not params: params = np.ones(featureNum)
		super(LinearHyp,self).__init__()
		self.parameters = params

	@staticmethod	#index
	def trans_x_Vector(x_1d):
		return x_1d

	def excution(self,xin):#support 2-d array in 
		return caculatepoly(self.parameters,xin)

	def __call__(self):
		transed_x = self.trans_x_Vector(xin)
		return self.excution(transed_x)



class regularGDF(train.factory):
	"""docstring for  regulation gradient descent"""
	def __init__(self,maxStep,regParam = 0.0,stRate=0.01,trainData=None,checkData=None):
		super(regularGDF,self).__init__(trainSet=trainData, checkData=checkData)
		self.stRate = stRate
		self.maxStep = maxStep
		self.regParam = regParam
	
	@staticmethod
	def iterExcRet(it,retExc,*argL,**argD):
		yin = it.trainData.y
		it.outDic['cost'] =  it.costFunc(yin,it.funOut)
		it.outDic['hyp'] = it.hyp
		if it.checkExc:	
			it.outDic['crossCost'] = it.costFunc(yin,it.hyp(it.checkData.x))
		return it.outDic

	def excFactory(self,it):
		it.trainMaxStep = self.maxStep
		it.stRate = self.stRate
		it.regParam = self.regParam
		x_expend = it.hyp.trans_x_Vector(it.trainData.x)
		it.funOut = it.hyp.excution(x_expend)
		yin = it.trainData.y
		dslen = len(it.trainData)		
		def excTraining(it):
			deltE = (yin-it.funOut)
			deltE.shape = (-1,1)
			deltPrar = (deltE*x_expend).sum(axis=0)/dslen
			regPers = (1-it.regParam*it.stRate/dslen)
			it.hyp.parameters = it.hyp.parameters*regPers + deltPrar*it.stRate
			it.funOut = it.hyp.excution(x_expend)
			return
		return excTraining

