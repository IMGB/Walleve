#logistic
from ..StandardImport import *

def  LogisticCostFun(funOut,yOut):
	costlist = yOut*np.log(funOut)+(1-yOut)*np.log(1-funOut)
	return costlist.sum()/len(yOut)

class LogisticHyp(hyp.hypothesis):
	"""docstring for """
	def __init__(self,numhyp,tureThreshold = 0.5,falseThreshold = 0.5):
		super(LogisticHyp,self).__init__()
		self.numhyp = numhyp
		self.tureThreshold = tureThreshold
		self.falseThreshold = falseThreshold
	
	def excution(self,xin): 
		retVal =  sigmoid(self.numhyp(xin))
		if retVal>self.tureThreshold:return True
		else :return False 


