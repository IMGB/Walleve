from ..StandardImport import *
import sci.stats as stats

class contiProbModle(hyp.hypothesis):
	"""docstring for probModle"""
	def calProb(self):
		return None

	def __init__(self,distMD = stats.norm):
		super(probModle, self).__init__()
		self.parameters = None
		self.rvMod = distMD
		self.rv = distMD
class MLET_tf(train.factory):
	"""docstring for MLETrainFac"""
	def __init__(self):
		super(MLET_tf, self).__init__()
		
	def excFactory(self,it):
		def subfun(it):
			return it.hyp.rv_conti.fit(it.trainData.trainData)
		return subfun
