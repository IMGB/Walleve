import numpy as np
import scipy as sci 
import matplotlib.pyplot as plt
import Classification.Classification as cf
import Regression.Regression as reg
import Trainer.trainer as train
import hypothesis.hypothesis as hyp
def caculatepoly(wpara,xin):
	return (wpara*xin).sum

def sigmoid(x):
	return 1/(1+np.exp(x))