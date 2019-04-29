#===========================================================================================
#
# This program contains routines required to generate predictions of the Stock Price
#
#===========================================================================================
import numpy as np


def predict_time_series(N_predict,N_nodes,x_data,y_data):
	'''
	Routine used to predict time series
	
	x_data: 
	y_data:
	'''
	x_predict = np.zeros(N_predict)
	y_predict = np.zeros(N_predict)
	
	
	# 1.a Scale the data 
	# 1.b construct the arrays for time series that will be used for training
	# 2.a Built the Neural network according to the number of nodes
	# 2.b Train the NN
	# 3. Generate predictions
	
	return x_predict, y_predict




def predict_stock_price(alpha,betaP,betaN,betaM,Pos,Neg,Market):
	'''
	Based on computed coefficients and positive/negative 
	sentiment signals and Market data,  we predict the return of
	the stock price using the linear model
	
	r(t) = alpha + beta_P*P+beta_N*N+betaM*M
	
	Since the return of a Stock is defined:
	r(t) = S(t)/S(t-1)-1
	
	Then compute the posterior distribution:
	S(t1) = (r(t1)+1)*S(t0) 
	     = (1+alpha + beta_P*P(t1)+beta_N*N(t1)+betaM*M(t1))*S(t0)
	
	Prediction Steps:
	0. Let us suppose we want to predict for T=t1,t2,...tMX
	1. Determine S(t0), which is the last known value of the stock price
	2. Compute the posterior distribution of S(t1) using Positive Sentiment Sig and Market Data
	3. Using Posterior Dist. of S(t1), compute S(t2)
	
	
	'''
	
	def model_return(ni,alpha,betaP,betaN,betaM,Pos,Neg,Market,N_samples):
		r=[]
		
		# Compute the sample of all returns
		for k in len(alpha):
			r_sample = alpha[k]+betaP[k]*Pos[ni]+betaN[k]*Neg[ni]+betaM[k]*Market[ni]
			r.append(r_sample)
		
		return r
		
	def model_stock_price(r,S_tm1):
		'''
		Compute the distribution of the Stock price 
		'''
		
		s=[]
		
		for i in range(r):
			for j in range(S_tm1):
				si = (1.0+r[i])*S_tmw[j] 
				s.append(si)
	
		return s

