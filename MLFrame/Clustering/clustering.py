from ..StandardImport import *
import numpy.random as nprand
def clusterCostFun(clusters,xSplice):
	aDis = calDistance(1,2)
	return aDis.flat.mean()

def xVector_cloasest_central_Indexs(xvectors,centrals):
	distArr = calDistance(xvectors,centrals)
	return distArr.argmin(axis = distArr.ndim-1)

class clusterClassify(hyp.hypothesis):
	"""docstring for clusterClassify"""
	def __init__(self,clusters):
		super(clusterClassify, self).__init__()
		self.clusters = np.array(clusters)
		self._mve = False

	def excution(self,xin):#support mutiple vector
		return xVector_cloasest_central_Indexs(xin, self.clusters)

class clusteringTrain(train.factory):
	def  __init__(self,trainData = None):
		super(clusteringTrain,self).__init__(costFun = clusterCostFun, trainData = trainData)
	@staticmethod
	def iterExcRet(it,retEx,*argL,**argD):
		exDic,cost = retEx 
		it.outDic['cost'] =  cost
		it.outDic['hyp'] = it.hyp
		it.outDic.update(exDic)
		return it.outDic

	def excFactory(self,it):
		def excTraining(it):
			lastCluster = it.hyp.clusters
			ClusterNum = len(it.hyp.clusters)
			xindexs =  np.apply_along_axis(xVector_cloasest_central_Indexs, 1,it.trainData,it.hyp.clusters)
			costArray = np.empty(ClusterNum)
			for i in range(ClusterNum):
				clus_arry = (it.trainData[xindexs==i]) 
				this_clu =  clus_arry.mean(axis=0)
				#print "clus:%d cluster:%s" %(i,this_clu)
				it.hyp.clusters[i] = this_clu
				costArray[i] = calDistance(this_clu, clus_arry).mean()

			return {'pointDiff':it.hyp.clusters-lastCluster} ,costArray.mean()
		return excTraining

class autoTrainCluster(clusteringTrain):
	def __init__(self,dataSet,repeatNum = 10,kStart = 2,clusterMinDif = 0.1):
		super(autoTrainCluster,self).__init__(trainData = dataSet)
		self.kNStart = kStart
		self.repeatNum = repeatNum
		self.clusterMinDif = clustetMinDif

	@staticmethod
	def trainOnce(kNum,minDif):
		NewCluster = (nprand.random(kNum,it.trainData.shape[-1])+minv)*(maxv-minv)
		itcus = mtrainFactor(hypothesis = clusterClassify(NewCluster),copyHyp = False)
		while 1:
			getDic = itcus.next()
			difAbs = getDic['pointDiff'].abs(axis = 0)
			if [difAbs<=minDif].all:break 
		return getDic

	def excFactory(self,it):
		it.clusterHyps = {}
		maxv = it.trainData.max(0) 
		minv = it.trainData.min(0)
		mtrainFactor  = clusteringTrain(it.dataSet) 
		it.repeatNum = repeatNum
		it.currentKNum = kStart
		
		def excTraining(it):
			costlist = []
			Diclist = []
			for i in range(it.repeatNum):
				theDic = self.trainOnce(it.currentKNum,it.clusterMinDif)
				clusterlist.append(theDic)
				costlist.append(theDic['cost'])
			retDic = Diclist[np.array(costlist).argmax()]
			retDic['kNum'] = it.currentKNum
			it.currentKNum += 1
			return retDic
		return excTraining
