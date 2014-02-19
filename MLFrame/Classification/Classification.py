#classification
from  .. import hypothesis.hypothesis as hyp
class Pattern(hyp.hypothesis):
	"""docstring for mode"""
	def __init__(self):
		super.__init__()
	
	
class PatternRecognition(object):
	"""docstring for PatternRecognition"""
	"""a class for classification"""
	def __init__(self,):
		super.__init__()
		self.modes = ()
		self.modeWeight = 1

	def getModeWeight(self,mode):
		return self.modeWeight

	def modeResultsort(self,restuls):
		return restuls.sort()

	def resultResultChoose(self,sortedResults):
		return sortedResults[0];

	def  __call__(self,xin):
		ResultList = [imode(xin) for imode in self.mode]
		self.modeResultsort(ResultList)
		

	

