from ..StandardImport import *
class svmClassify(hyp.hypothesis):
	def __init__(self,kernal = None):
		super(svmClassify,self).__init__()
		self.kernal = kernal
		self.support_x = None
		self.support_y = None
		self.multiplier = None
		self.b = 0
		

	def excution(self,xin):
		kernalResult =  np.apply_along_axis(self.kernal, 1, self.support_x,xin)
		return (self.multiplier*kernalResult*self.support_y).sum()+self.b
		
	def decorateOut(self,excout):
		return sgn(excout)


class svm_smo_fc(train.factory):
	"""docstring for SVM_SMO"""
	def __init__(self,slack,kkt_toler=0.0001,trainData=None,costFun = None):
		super(svm_smo_fc, self).__init__(trainData,costFun)
		self.slack = slack
		self.kkt_toler = kkt_toler
		self.costFunc = None

	def excFactory(self,it):

		it.hyp.multiplier = np.zeros(len(it.trainData))
		#it.hyp.multiplier = np.zeros(len(it.trainData))
		it.hyp.b = np.random.rand()*self.slack
		it.slack  = self.slack
		xs = it.trainData.x
		ys = it.trainData.y
		h = it.hyp
		h.support_x = xs
		h.support_y = ys
		it.errs = np.apply_along_axis(lambda xin:h.excution(xin),1, arr=xs).reshape(-1) - ys
		it.kkt_toler = self.kkt_toler
		print "errs is %s"%it.errs
		def multiplier_cal(i,j,Ei,Ej):
			eta = h.kernal(xs[i],xs[i])+h.kernal(xs[j],xs[j])- 2.0*h.kernal(xs[i],xs[j])
			C = it.slack
			if ys[i]!=ys[j]:
				L = max(0,h.multiplier[j]-h.multiplier[i])
				H = min(C,C+h.multiplier[j]-h.multiplier[i])
			else:
				L = max(0,h.multiplier[j]+h.multiplier[i] - C)
				H = min(C,h.multiplier[j]+h.multiplier[i])
		 	old_mult_j = h.multiplier[j]
		 	new_mult_j = h.multiplier[j] + ys[j]*(Ei-Ej)/eta

		 	if new_mult_j>H:
		 		new_mult_j = H
		 	elif new_mult_j<L:
		 		new_mult_j = L
		 	new_mult_i = h.multiplier[i] + ys[i]*ys[j]*(old_mult_j-new_mult_j)
		 	return (new_mult_i,new_mult_j)

		def excutInLoop(it,mult_i):
			C = it.slack
			# find j as |E1-E2| max
			if it.errs[mult_i]>0:
				index = it.errs.argmin()
			else:	
				index = it.errs.argmax()

			if index != mult_i:
				new_mult_i,new_mult_j = multiplier_cal(mult_i, index, it.errs[mult_i], it.errs[index])
				if abs(new_mult_j-h.multiplier[index])>0.00001:
					return new_mult_i,new_mult_j,mult_i,index
			#find j in no bound
			no_bound_indexs = np.argwhere((h.multiplier>0) & (h.multiplier<C)).reshape(-1)
			for index in np.random.permutation(no_bound_indexs):
				if index == mult_i:continue
				new_mult_i,new_mult_j = multiplier_cal(mult_i, index, it.errs[mult_i], it.errs[index])
				if abs(new_mult_j-h.multiplier[index])>0.00001:
					return new_mult_i,new_mult_j,mult_i,index
			#find j in all trainset 
			bound_indexs = np.argwhere((h.multiplier==0) | (h.multiplier==C)).reshape(-1)
			for index in np.random.permutation(bound_indexs):
				if index == mult_i:continue
				new_mult_i,new_mult_j = multiplier_cal(mult_i, index, it.errs[mult_i], it.errs[index])
				if abs(new_mult_j-h.multiplier[index])>0.00001:
					return new_mult_i,new_mult_j,mult_i,index
			return None

		def choseFromOutLoop(it):
			## satisfy KKT condition  
    			# 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)  
    			# 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)  
    			# 3) yi*f(i) <= 1 and alpha == C (between the boundary)  
    			## violate KKT condition  
    			# because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so  
    			# 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct)   
    			# 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)  
    			# 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized  
			toler = it.kkt_toler
			C = it.slack
			gs = ys*it.errs
			no_bound_arg =np.argwhere((h.multiplier>0) & (h.multiplier<C)).reshape(-1)
			if len(no_bound_arg)>0:
				no_bound_Mult = np.absolute(gs[no_bound_arg])-toler
				relative_arg = np.argsort(no_bound_Mult)
				for i_inx in relative_arg[::-1]:
					if no_bound_Mult[i_inx]>0:
						mult_i = no_bound_arg[i_inx]
						ret = excutInLoop(it,mult_i)
						if ret:return ret
						else:continue
					else:break
			'''
			no non_bound data that violate the kkt
			now we want to find the bound data that violate the kkt
			two situation (gs = y[i]*E_i)
			y[i]*E_i >= -toler alpha = 0 violate situation:gs<-toler vio_va =-toler-gs
			y[i]*E_i <= toler alpha = C violate situation:gs>toler vio_va = gs-toler
			so unify_vio_v = gs*(alpha/C*2-1)-t
			'''
			unify_vio = gs*(h.multiplier/C*2-1)-toler
			sortedArg = np.argsort(unify_vio)

			for i_inx in sortedArg[::-1]:
				if unify_vio[i_inx]>0:
					mult_i = unify_vio[i_inx]
					ret = excutInLoop(it,mult_i)
					if ret:
						return ret
					else:continue
				else:
					raise train.TrainEndSuc("no violated multiplier")
			else:#no alpha i is 
				return None

		def excTraining(it):
			C = it.slack
			ret = choseFromOutLoop(it)
			if not ret:
				raise train.TrainError("no i and j can improve positive")
				return
			else:
				new_mult_i,new_mult_j,i,j = ret

			old_mult_i = h.multiplier[i]
			old_mult_j = h.multiplier[j]
			h.multiplier[i] = new_mult_i
		 	h.multiplier[j] = new_mult_j
		 	Kij = h.kernal(xs[j],xs[i])
		 	b_i = h.b-it.errs[i]-ys[i]*h.kernal(xs[i],xs[i])-ys[j]*Kij
		 	b_j = h.b-it.errs[j]-ys[i]*Kij-ys[j]*h.kernal(xs[j],xs[j])
		 	if new_mult_i>0 and new_mult_i<C:
		 		h.b = b_i
		 	elif new_mult_i>0 and new_mult_i<C:
		 		h.b = b_j
		 	else:
		 		h.b = (b_j+b_i)/2.0
		 	oldErrs = it.errs
		 	newRes =np.apply_along_axis(lambda xin:h.excution(xin),1, arr=xs).reshape(-1)
		 	it.errs = newRes - ys

		 	print "chose alpha i:%s j:%s" %(i,j)
		 	print "alpha is %s" %h.multiplier
		 	print "b is %s"%h.b
		 	print "errs is %s"%it.errs
		 	print "delt errs is %s" %(it.errs-oldErrs)
			print "alpha*y is %s" %(h.multiplier*ys).sum()
			return None
		return excTraining