import numpy as np
import scipy as sci 
import matplotlib.pyplot as plt
import mysort

def getDistance(vectorA,vectorB):
	return ((vectorB-vectorA)**2).sum;

def sortDistance(Distancelist,index):
	least_K_list = [0]*5

	for dist in dits_s
		least_K_list.append(dist)
		least_K_list = sorted(least_K_list)[:5]
	return least_K_list

def getMostCloseFun(DistanceList):
	mDic = {}
	for xclose,yclose,distances in DistanceList:
		if yclose in mDic:
			mDic[yclose] = 1
		else:
			mDic[yclose]++
	returnFun = None
	maxNum = 0
	for tpair in mDic.items():
		if tpair[1]>maxNum:
			returnFun = tpair[0] 
			maxNum =  tpair[1]
	return returnFun


def excuteKNN(trainset,xq,k=3):
	if k<x.ndarray.shape[0]:
		k=x.ndarray.shape[0]
	if k<3:	
		return
	distanceAndIndexList = sortDistance(getDistance(trainset[:,0], xq)]

	trainAndDistList = np.hstack((trainset[distanceAndIndexList[:,0]] , distanceAndIndexList))

	getMostCloseFun(trainAndDistList);
