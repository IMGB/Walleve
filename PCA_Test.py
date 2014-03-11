from MLFrame.PCA import*
'''
num = 5
a = np.random.rand(num)*5

x1 = a.reshape(-1,1)
x2 = (a*2+np.random.rand(num)).reshape(-1,1) +2
x3 = x1+x2
xset = np.hstack((x1,x2,x3))
'''
xset = np.array([[-1,-2],[-1,0],[0,0],[2,1],[0,1]])
print xset



plt.scatter(xset[:,0],xset[:,1])
plt.show()

fun =  pca_k(xset, 1)
print "result is %s" %fun(xset)
