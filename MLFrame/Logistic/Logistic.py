#logistic
from ..StandardImport import *


class LogisticHyp(hyp.hypothesis):
	"""docstring for """
	def __init__(self, featureNum=1):
		super.__init__(1)
		self.featureNum = featureNum
		self.parameters = np.zeros(featureNum)

		

def funWithPara(paraVector,xin):
	sigmoid(caculatepoly(paraVector,xin))
	return

def  squareDValue(a,b):
	return ((a - b)**2).sum()/len(a))


def trainAlgrithom():
	for returnPara,funOut,disable_Restrain in GD.gradinet_DescentByStep(trainingset,Func,parameters) :
		print costFunctionWithOut(trainingSet,funOut)

