import numpy as np
class dataSetExp(Exception):
	pass

class ds_2(object):
	"""docstring for dataSetPair"""
	def __init__(self,pairSet = None):
		self.pairSet = pairSet

	def __iter__(self):
		return iter(self.pairSet)

	def addXYPair(self,x,y):
		return dataSetPair(self.pairSet.append((x,y)))

	def addXYPair(self,pair):
		return dataSetPair(self.pairSet.append((x,y)))

	def __len__(self):
		return len(pairSet)
	
	def __nonzero__(self):
		if len(self._x)>0:
			return True
		else:
			return False

	@property
	def x(self):
		return [xx for xx,yy in self.pairSet]

	@property
	def y(self):
		return [yy for xx,yy in self.pairSet]
	
class dataSetNpPairIter(object):
	def __init__(self,x,y=None):
		self.xit = iter(x)
		self.yit = iter(y)

	def next(self):
		return self.xit.next(),self.yit.next()


class ds_2_np(object):
	"""docstring for ds_2_np,instance of this class can not be changed,
	but not copy the buffer for performance situation"""
	def __init__(self,x,y):
		if x!=None and y!=None and len(y)!=len(x): 
			raise TypeError("x and y are not adaptive with each other")
		elif len(x)<1 or len(y)<1:
			raise TypeError("len of x or y is less than 1")
		else:	
			self._x = x
			self._y = y

	@property
	def x(self):
		return self._x

	@property
	def y(self):
		return self._y

	def __str__(self):
		return ''


	def __add__(self,dataset):
		if dataset.x and self._x: newx = np.concatenate((self._x,dataset.y))
		if dataset.y and self._y: newy = np.concatenate((self._x,dataset.y))
		return self.__class__(newx,newy)

	def __iter__(self):
		return dataSetNpPairIter(self._x,self._y)

	def __len__(self):
		return len(self._x)

	def __getitem__(self,index):
		return self.__class__(self.x[index], self.y[index])

	def plot(self,x_splice):
		pass

'''	
	def __add__(self,dataset):
		if dataset.x and self.x: newx = np.append(self.x,dataset.x, axis=0)
		if dataset.y and self.y: newy = np.append(self.x,dataset.x, axis=0)
		return dataSetPairNp(newx,newy)
'''

class ds_2_np_patter(ds_2_np):
	''' x is absolutly np array and y out is a key represent a pattern'''
	def __init__(self,x,y):
		super(ds_2_np_patter,self).__init__(x,y)
		self.patternDic = self.classify_Pattern()
	'''
	def classify_Pattern(self):
		patterns = {}
		u, indices = np.unique(self._y, return_index=True)
		for xtype,inx in zip(u,indices):
			patterns[xtype] = self._x[indices]
		return patterns
		def classify_Pattern(self):
		patterns = {}
		self.patt_y, indices = np.unique(self._y, return_index=True)
		self.patt_x = self._x[indices]
		return 
	
	def xOfPattern(self,patternKey):
		return self.patt_x[self.patt_x == patternKey]

	def cal_Pattern_Prob(self,patternKey):
		return float(len(self.patt_x[self.patt_x == patternKey]))/len(self._y) 
	'''
	def classify_Pattern(self):
		patterns = {}
		tempX = self._x
		tempY = self._y
		while len(tempY)>0:
			y_tp = tempY[0]
			patterns[y_tp] = tempX[tempY==y_tp]
			tempX = tempX[tempY!=y_tp]
			tempY = tempY[tempY!=y_tp]
		return patterns
	
	def xOfPattern(self,patternKey):
		if patternKey in self.patternDic:
			return self.patternDic[patternKey]
		else :return None

	def cal_Pattern_Prob(self,patternKey):
		if patternKey in self.patternDic:
			return float(len(self.patternDic[patternKey]))/len(self._y) 
		else:
			return None


	