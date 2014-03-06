from ..StandardImport import *
from ..LinearRegression import LinearReg as LR
from ..Trainer import gradDescTrainer as gD
class KNNParent(hyp.hypothesis):
	"""docstring for KNN"""
	def __init__(self, korder=0,baseData = None):
		super(KNNParent, self).__init__()
		self.korder= korder
		self.baseData = baseData

	@staticmethod
	def choseKPoint(DataSet,k,xq):
		aDistance = calDistance(DataSet.x, xq)
		retInd = np.argsort(aDistance)[:k]
		return dSet.dataSetPatY(DataSet.x[retInd], DataSet.y[retInd])

	@staticmethod
	def funOut(weights,kDataSet,xin):
		return None

	@staticmethod
	def cal_Weights(kDataSet):
		return kDataSet.sqrDistance**(-1)

	def excution(self,xin,korder = None): #not suppot mutiple vector
		if not korder:korder = self.korder
		kDataSet = self.choseKPoint(self.baseData,self.korder,xin)
		weights = self.cal_Weights(kDataSet) #kDataSet.sqrDistance**(-1)	
		return self.funOut(weights, kDataSet,xin)


class KNNClassify(KNNParent):
	@staticmethod
	def funOut(weights,kDataSet,xin):
		sortDic = {}
		[ sortDic[v] = (weights*diffFun(v, kDataSet.y)).sum() for v in kDataSet.patternDic.keys]	
		return sorted(sortDic)[-1]


class KNNRegress(KNNParent):
	@staticmethod
	def funOut(weights,kDataSet,xin):
		return (weights*kDataSet.y).sum()/weights.sum()


class LOESS(KNNParent):
	def __init__(self, trainFac,hyp,korder=0,baseData = None):
		super(LOESS, self).__init__()
		self.trainFac = trainFac
		self.hyp = hyp

	@staticmethod
	def funOut(weights,kDataSet,xin):
		it = self.trainFac(self.hyp,kDataSet)
		thehyp = it(debug=False)['hyp']
		return thehyp(xin)

class Linear_gd_LOESS(LOESS):
	"""docstring for Linear_gd_LOESS"""
	def __init__(self,korder=0,baseData = None):
		thehyp = LR.LinearHyp()
		theFac = gD.gradDesFactory(1000)
		super(Linear_gd_LOESS, self).__init__(theFac,thehyp,korder,baseData)

	@property
	def featureNum(self):
	    return len(self.hyp.parameters)
	@featureNum.setter
	def featureNum(self, value):
	    self.hyp.parameters = np.zeros(value)
	
