from .StandardImport import *
def pca_scalar_x(r_max,xin):# the main will be set to 0
	mean = xin.mean(axis=0)
	mean_0_x = xin - mean
	factor = r_max/np.amax(np.absolute(mean_0_x),axis=0)
	return mean_0_x*factor,factor,mean

def pca_data(xin):
	x_s,scal_factor,mean = pca_scalar_x(1,xin)
	trans = x_s.transpose();
	covariation = np.dot(trans,x_s)/len(x_s)
	eigVals,eigVects = np.linalg.eigh(covariation)
	return eigVals,eigVects.T,mean,scal_factor

def produceXMap(mean,factor,basis_vectors):
	def x_map(xin):
		x_s = (xin - mean)*factor
		if x_s.ndim < 2:x_s.shape = (1,-1)
		return (np.dot(x_s,basis_vectors.transpose()))
	return x_map

def pca_k(xin,k):
	eigVals,eigVects,mean,scal_factor = pca_data(xin)
	eigArg =  eigVals.argsort()
	basis_vector_index  = eigArg[::-1][:k]
	basis_vector = eigVects[basis_vector_index]SVM
	return produceXMap(mean, scal_factor, basis_vector)

def pca_auto(xin,varaint_loss):
	eigVals,eigVects,mean,scal_factor = pca_data(xin)
	threshold = 1-varaint_loss

	eigArg =  eigVals.argsort()
	eigVals = eigVals[eigArg]
	eigVects = eigVects[eigArg]
	k=0
	for val in eigVals[::-1]:
		eigValsum += val
		k +=1
		if eigValsum>threshold:
			break
		else:continue
	basis_vector = eigVects[::-1][:k]
	return produceXMap(mean, scal_factor, basis_vector)