from MLFrame.StandardImport import *
from MLFrame.LinearRegression import LinearReg as LR
#Linear regression z = 3x**2+4xy+5y**2+7

try:  # do not edit! added by PythonBreakpoints
    from ipdb import set_trace as _breakpoint
except ImportError:
    from pdb import set_trace as _breakpoint

xset = np.random.rand(100,2)


def testDataCreate1(num):
	x = xset[:,0]
	y = xset[:,1]
	zset = 3*x**2+4*x*y+5*y**2+7
	return dSet.ds_2_np(xset, zset)


def testDataCreate2(num):
	x = xset[:,0]
	y = xset[:,1]
	T1 = (x**2).reshape((-1,1))
	T2 = (x*y).reshape((-1,1))
	T3 = (y**2).reshape((-1,1))
	T4 = np.ones(len(xset)).reshape((-1,1)) 
	zset = 3*T1+4*T2+5*T3+7*T4
	xin = np.hstack((T1,T2,T3,T4))
	return dSet.ds_2_np(xin, zset.reshape(-1))

def _testFrom(x_1d):
	if x_1d.ndim == 1:
		x = x_1d[0]
		y = x_1d[1]
		c = np.ones(1)
	elif x_1d.ndim == 2:
		x = x_1d[:,0]
		x.shape = (-1,1)
		y = x_1d[:,1]
		y.shape = (-1,1)
		c = np.ones(len(x))
		c.shape = (-1,1)
	else :
		raise TypeError(" not support dimension over 2")	
	ret = np.hstack((x**2,x*y,y**2,c))
	#print "new set is %s" %ret
	return 	ret

mTF = LR.regularGDF(150000,stRate = 0.001)

testHyp1 = LR.LinearHyp()
testHyp1.trans_x_Vector = _testFrom
testHyp1.parameters = np.ones(4)
trainSet1 = testDataCreate1(100)
it1 = mTF(testHyp1,trainSet1,copyHyp = False)
ret1 =  it1()
print "1 finished"
print (ret1['hyp'].parameters,ret1['cost'],ret1['step'] )

testHyp2 = LR.LinearHyp()
testHyp2.parameters = np.ones(4)
trainSet2 = testDataCreate2(100)
it2 = mTF(testHyp2,trainSet2,copyHyp = False)
ret2 =  it2()
print "2 finished"
print (ret2['hyp'].parameters,ret2['cost'],ret2['step'] )

'''
for retDic in mTF(testHyp,copyHyp = False,checkExc = True):
	print (retDic['hyp'].parameters,retDic['cost'],retDic['crossCost'] ,retDic['step'] )
'''



















