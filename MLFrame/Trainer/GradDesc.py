#GRADIENT-DESCENT
def  gradinet_DescentByStep(trainingset,Func,params,step,MaxStep=10000) :
	disable_Restrain = False
	returnPara = params.copy()
	while  not disable_Restrain :
		funOut = Func(params,trainingset[:,0]
		deltparam = -1*trainingset[:,0]*(trainingset[:,1] - funOut)
		ttldelt = deltparam.sum()*step
		returnPara += ttldelt
		if MaxStep-- <= 0:
			disable_Restrain = True
		yield returnPara,funOut,disable_Restrain
	return 

def  gradinet_DescentApproxiByStep(trainingset,Func,params,step,MaxStep=10000):
	disable_Restrain = False
	returnPara = params.copy()
	while  not disable_Restrain:
		deltparam = -1*trainingset[:,0]*(trainingset[:,1]-Func(params,trainingset[:,0])
		Ttldelt = deltparam*step
		returnPara += Ttldelt
		MaxStep--;
		if MaxStep <= 0:
			disable_Restrain = True
		yield returnPara,disable_Restrain

