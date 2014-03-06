from ..customerFun import *
import copy
ex_type_end_unknow = 0
ex_type_max_step = 1
ex_type_excut_end = 2
ex_type_excut_error = 3
ex_type_init_error = 4


trianExcStr = (	'train stop for unknown reason:',
		'train reach the max step:',
		'trainer exction ended correctly:',
		'trainer exction ended for error:'
		'trainer exction init error:')

class StopTraining(StopIteration):
	def __init__(self,Des="",errType=0,hyp = None):
		super(StopTraining,self).__init__(trianExcStr[errType]+Des)
		self.ex_type = errType
		self.hyp = hyp

class TrainEndSuc(StopTraining):
	def __init__(self,Des="",errType=2,hyp = None):
		super(TrainEndSuc,self).__init__(Des,errType,hyp)

class TrainError(StopTraining):
	pass

class trainer(object):
	def __init__(self,trainData,hyp,desc=None,costFun=None,checkExc = False,maxStep = 0):
		super(trainer,self).__init__()
		self._trainStep = 0
		self._desc = desc
		self._excution = None
		self.trainMaxStep = maxStep
		self._trainData = trainData
		self.costFunc = costFun
		self.outDic = {}
		self.checkExc = checkExc
		self._hyp = hyp
		self.checkData = None

	def __iter__(self):
		return self

	def next(self):
		if self.trainMaxStep and (self._trainStep >= self.trainMaxStep):
			raise TrainEndSuc(errType=ex_type_max_step)
		retExc = self._excution(self)
		self.iterExcRet(self,retExc)
		self._trainStep += 1
		self.outDic['step'] = self._trainStep
		return  self.outDic

	def __call__(self,debug=False):#continue training until train end successfully
		try:
			if debug:
				while 1:print self.next()
			else:
				while 1:self.next()
		except TrainEndSuc:
			pass
		return self.outDic
	@property
	def hyp(self):
		return self._hyp

	@property
	def trainData(self):
	 	return self._trainData

	@property
	def excution(self):
	 	return self._excution

	@excution.setter
	def excution(self, value):
		if self._excution: raise AttributeError("excution has been setted")
	 	else:	self._excution = value

class factory(object):
	def __init__(self,trainData=None,costFun = None,checkData = None):
		if costFun :	self.costFunc = costFun
		else :self.costFunc = None
		self.trainData = trainData
		self.checkData = checkData
		self.trainClass = trainer

	def trainerDesc(self):
		return  "row trainer"

	def excFactory(self,it):
		def excTraining(it):
			return None
		return excTraining

	@staticmethod
	def iterExcRet(it,retExc,*argL,**argD):
		yin = it.trainData.y
		if it.costFunc: 	it.outDic['cost'] =  it.costFunc(yin,it.hyp(it.trainData.x))
		it.outDic['hyp'] = it.hyp
		if retExc: it.outDic.append(retExc)
		if it.checkExc:	it.outDic['crossCost'] = it.costFunc(yin,it.hyp(it.checkData.x))
		return it.outDic


	def iterFactory(self,hypothesis,trainData,checkData,checkExc):
		iterator = self.trainClass(trainData,hypothesis,self.trainerDesc,self.costFunc,checkExc)
		iterator.checkData = self.checkData
		iterator.excution = self.excFactory(iterator)
		iterator.iterExcRet = self.iterExcRet
		return  iterator

	def __call__(self,hypothesis,trainData = None,copyHyp = False,checkExc = False,checkData = None,**arg):
		if trainData==None:		trainData = self.trainData
		if checkData==None:		checkData = self.checkData

		if trainData==None: 
			raise TrainError(ex_type_excut_error,"no traindata for training") 
		elif not hypothesis:
			raise TrainError(ex_type_excut_error,"no hypothesis for training")
		elif checkData==None and checkExc:
			raise TrainError(ex_type_excut_error,"no checkData for training")
		else: 
			if copyHyp:
				return self.iterFactory(copy.deepcopy(hypothesis),trainData,checkData,checkExc)
			else:
				return self.iterFactory(hypothesis,trainData,checkData,checkExc)
