#hypothesis
class hypothesis(object):
	"""docstring for hypothesis"""
	def __init__(self,params=None):
		super.__init__()
		self.parameters = params
	def excution(self,params):
		pass
	def __call__(self,xin):
		self.excution(self.parameters)
