from ..StandardImport import *
import copy
class geneticFac(train.factory):
	def __init__(self,fitness,threahold,mutation,popu_Size,mut_Rate=0.5,crossRate=0.5):
		super(geneticFac,self).__init__()
		self.fitnessFun = fitness
		self.popu_Size = popu_Size
		self.mut_Rate = mut_Rate
		self.crossRate = self.crossRate
		self.mutation = mutation
		self.threahold = threahold
	@staticmethod
	def generate(poplist):
		return None




	def excFactory(self,it):
		fitness = self.fitnessFun
		mutation = self.mutation
		it.threahold = self.threahold
		it.crossRate = self.crossRate
		it.mut_Rate = self.mut_Rate
		popsize = len(it.population)
		it.population = [self.mutation(it.hyp) for i in range(self.popu_Size)]
		it.fitnessArr = np.array([fitness(hyp) for hyp in it.population])
		def pickSampleUsingProb(problist,k):
			retArray = []
			randfactor = np.random.rand(k)
			curpoint = 0
			sumover = problist[0]
			for factor in randfactor:
				while factor<=sumover:
					curpoint++
					sumover += problist[curpoint]
				else: 
					retArray.append(curpoint)
			return retArray

		def excTraining(it):
			if fitnessArr.amax() <=threahold:
				fitProb = np.true_divide(fitnessArr , fitnessArr.sum())
				new_Population = [it.population[index] for index in pickSampleUsingProb(it.fitProb,int((1-it.crossrate)*popsize))]
				generateList = [it.population[index] for index in pickSampleUsingProb(it.fitProb,int(it.crossrate*popsize))]
				new_Population += generate(generateList)
				[mutation(new_Population[index]) for index in np.random.randint(int(it.mut_Rate*popsize))]
				it.population = new_Population
				it.fitnessArr = np.array([fitness(hyp) for hyp in it.population])
				return { it.population}
			else:
				raise train.TrainEndSuc(Des="reach threahold")

		return excTraining


