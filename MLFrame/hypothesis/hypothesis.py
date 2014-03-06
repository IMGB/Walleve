#hypothesis
import numpy as np
import matplotlib.pyplot as plt
class hypError(Exception):
	"""docstring for hypError"""
	def __init__(self, desc):
		super(hypError, self).__init__(desc)

class HypInputError(hypError):
	pass

class HypExcutionError(hypError):
	pass
		

class hypothesis(object):
	"""docstring for hypothesis"""
	def __init__(self,params=None):
		self.parameters = params
		self._mve = True #support mutiple vector exction

	@property
	def support_mve(self):
		return self._mve

	def plot(self,plotRange=None):
		return None

	def excution(self,xin):
		return None

	def checkXin(self,xin):
		return True
	## important
	##!! if xin.ndim > 1 i treat xin as mutiply xin and if xin.ndim = 1 we think xin a single vector in put 
	def __call__(self,xin): 
		if not self._mve and xin.ndim>1:
			raise HypInputError("not support the input shape size")
		else:
			try:
				if not self.checkXin(xin):
					raise HypInputError("not correct putin")
			except Exception, e:
				raise HypExcutionError("excution error")
			return self.excution(xin)
	
	


