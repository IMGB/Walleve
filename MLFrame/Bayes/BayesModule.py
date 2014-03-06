from ..StandardImport import *
class Pattern_class(object):
	"""docstring for mode"""
	def __init__(self,key):
		super(Pattern_class,self).__init__()
		self.key = key

class bayesMap(hyp.hypothesis):
	def __init__(self,patterns = None):
		super(bayesMap,self).__init__()
		self.patterns = patterns
		self.weight = 1
		self._mve = False
	
	def prior_prob(self,pattern):
		return None

	def similarity(self,pattern,xin):
		return None

	def excution(self,xin,retProb=False):
		posterior_prob = np.zeros(len(self.patterns))
		for i in range(len(self.patterns)):
			pattern = self.patterns[i]
			prior_prob = self.prior_prob(pattern)
			similarity = self.similarity(pattern,xin)
			posterior_prob[i] =prior_prob*similarity
			
		if retProb:
			return zip(self.patterns,posterior_prob)
		else:
			return self.patterns[posterior_prob.argmax()].key





		