#SVMTest
# ready data
from MLFrame.StandardImport import *
from MLFrame.SVM.svm import * 
x = np.random.rand(30)*10-5
y = np.random.rand(30)*10-5
#y = 2.5*x+3

indecs = np.absolute(y-(2.5*x+3))>2
x = x[indecs]
y = y[indecs]

xsup = np.random.rand(3)*10-5
ysup1 = 2.5*xsup+3+2.1
ysup2 = 2.5*xsup+3-2.1

x = np.hstack((x,xsup,xsup))
y = np.hstack((y,ysup1,ysup2))
z = ((2*y-5*x-6)>0)*1.0*2.0-1.0

tdset = dSet.ds_2_np(np.hstack((x.reshape(-1,1),y.reshape(-1,1))), z)

pltx =  np.arange(-6,6)

print tdset.x
print tdset.y
plt.plot(pltx,2.5*pltx+3,c='g')
plt.scatter(tdset.x[tdset.y==1.,0], tdset.x[tdset.y==1.,1],c="b")
plt.scatter(tdset.x[tdset.y==-1.,0], tdset.x[tdset.y==-1.,1],c="r")

#  ready for svm training
def linearKernal(x1,x2):
	return (x1*x2).sum()
svmc = svmClassify(linearKernal)

svmfc = svm_smo_fc(slack=10,kkt_toler=0.0001,trainData = tdset)

it = svmfc(svmc)
while 1:
	raw_input("continue>>")
	it.next()
	pass


