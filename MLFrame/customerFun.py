import numpy as np
def caculatepoly(wpara,xin):
	retArray = wpara*xin
	return retArray.sum(axis=retArray.ndim-1)

def sigmoid(x):
	return 1/(1+np.exp(-1*x))

def squareDValue(a,b):
	retArr = (a-b)**2
	return retArr.sum(axis=retArr.ndim-1)

def defaultCostfun(funout,yout):
	return squareDValue(funout,yout)/len(yout)

def calDistance(vectorA,vectorB):
	temp = (vectorB-vectorA)**2 
	return temp.sum(axis=temp.ndim-1);

def diffFun(A,B):
	return A==B