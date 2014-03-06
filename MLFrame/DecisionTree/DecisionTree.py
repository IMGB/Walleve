#DecsisonTree
from ..StandardImport import *
def dTreeCostFun(yOut,funOut):
	return (yOut - funOut)/len(yOut)

class treeNode(dict):
	def __init__(self,pathFromType):
		super(treeNode,self).__init__()
		self.attrKey = None
		self.pathFromType = pathFromType
		#self.attrType = attrType
		self.outValue = None

	def leaf(self):
		return self.outValue != None


class DTree(hyp.hypothesis):
	"""docstring for DTree"""
	def __init__(self):
		super(DTree, self).__init__()
		self.root = None
		self._mve = False

	def excution(self,xin):
		node = self.root
		while 1:
			xkey = node.attrKey
			node = node[xin[xkey]]
			if node.leaf():
				return node.outValue

def cal_entropy(trainData):
	theOuts = trainData.patternDic.keys()
	probs = np.array([trainData.cal_Pattern_Prob(key) for key in theOuts])
	return (-1*probs*np.log2(probs)).sum()

def mostGeneralPattern(trainSet):
	maxkey = None
	maxProb = 0
	for key in trainSet.patternDic:
		tProb = trainSet.cal_Pattern_Prob(key)
		if maxProb >= tProb:
			maxkey = key
			maxProb = tProb
	return maxkey

def cal_gain(trainData,attrKey):
	ttlSize = len(trainData)
	xin = trainData.x
	yin = trainData.y
	subset = {}
	while len(xin)>0:
		xtype = xin[0,attrKey] 
		trueIndex = (xin[:,attrKey] == xtype)
		subset[xtype] = dSet.ds_2_np_patter(xin[trueIndex],yin[trueIndex])
		falseIndex = (xin[:,attrKey] != xtype)
		xin = xin[falseIndex]
		yin = yin[falseIndex]
	entropylist = [cal_entropy(subData)*len(subData)/ttlSize for subData in subset.values()]
	gain = cal_entropy(trainData) - sum(entropylist)
	return gain,subset

def trainNode(stack):
	if len(stack)==0:
		raise train.TrainEndSuc("no left in stack")
	(trainedNode,datas,attrs) = stack.pop()
	if len(attrs) == 0:
		trainedNode.outValue = mostGeneralPattern(datas)
		return
	elif len(datas.patternDic)==1:
		trainedNode.outValue = datas.patternDic.keys()[0]
		return
	maxGains = (0,None,-1)
	for attrKey in attrs:
		gain, subdata = cal_gain(datas,attrKey)
		if gain>=maxGains[0]:
			maxGains = (gain,subdata,attrKey)

	attrs.remove(maxGains[2])
	trainedNode.attrKey = maxGains[2]
	
	for attrType in maxGains[1]:
		newNode = treeNode(attrType)
		trainedNode[attrType] = newNode
		stack.append((newNode,maxGains[1][attrType],list(attrs)))
	return  trainedNode

class ID3(train.factory):
	"""docstring for ID3"""
	def __init__(self,trainData=None):
		super(ID3, self).__init__(trainData=trainData)
	
	def excFactory(self,it):
		attrs = list(range(it.trainData.x.shape[1]))
		it.hyp.root = treeNode('root no path')
		trainStack = [(it.hyp.root,it.trainData,attrs),]
		def excTraining(it):
			trainNode(trainStack)
		return excTraining

