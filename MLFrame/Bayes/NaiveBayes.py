from ..StandardImport import *
from .BayesModule import*
class naiveBayes(bayesMap):
	def __init__(self,trainData = None):
		super(naiveBayes,self).__init__()
		if trainData:
			self.trainNB(trainData)
		self.equiPara = 0
		self.patterns = None


	def trainNB(self,trainData):
		e = self.equiPara
		def classifyXpara(xset):
			xin = xset
			dlen = len(xset)
			xparaDic = {}
			while len(xin)>0:
			 	xtype = xin[0]
				xparaDic[xtype] = len(xin[xin==xtype])
				xin = xin[xin!=xtype]
			p = 1.0/len(xparaDic)
			retDic = {}
			for key,nc in xparaDic.items():
				retDic[key] = (nc+e*p) /(dlen+e)
			return retDic
		self.patterns = []

		for key,datas in trainData.patternDic.items():
			if datas.ndim == 1:datas = reshape((-1,1))
			arrayList = [None,]*datas.shape[1]
			for i in range(datas.shape[1]):
				arrayList[i] = classifyXpara(datas[:,i])
			newPattern = Pattern_class(key)
			newPattern.parameters = arrayList
			newPattern.prior_prob = trainData.cal_Pattern_Prob(key)
			self.patterns.append(newPattern)
		return None

	def prior_prob(self,pattern):
		return pattern.prior_prob

	def similarity(self,pattern,xin):
		retProb = 1
		for i in range(len(xin)):
			patterDic =pattern.parameters[i]
			if xin[i] in patterDic:
				retProb *= patterDic[xin[i]]
			else:
				retProb = 0
				break
		return retProb
