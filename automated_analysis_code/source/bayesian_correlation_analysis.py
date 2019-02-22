import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler
from source.class_vehicle_data import Vehicle_data
import os
import emcee
import scipy.optimize as op
import corner


def compute_correlation_posteriors(pos_scale,neg_scale,sentiment_curve_scaling,sentiment_scaling_Xmin,sentiment_scaling_Xmax, remove_outliers,target_car,thresholds,N_conv_min,N_conv_max,N_mcmc_burn,N_mcmc_walkers,N_mcmc_runs,N_correlation_samples):
	'''
	This code will generate the Bayesian correlations
	'''
	
	# Geneare the list of folders
	output = ["output_t"+str(int(100*threshold)) for threshold in thresholds]
	
	# Initialize all car data
	all_cars = [Vehicle_data() for k in range(len(output))]
	
	b1_scale = pos_scale#1.0e-3
	b2_scale = neg_scale#1.0e-3
	
	# Create a new directoy for the target vehicle
	new_folder="data/"+target_car
	
	try:
		os.mkdir(new_folder)
	except FileExistsError:
		print('Directory: ', new_folder, ' already exists')
	
	
	plt.clf()
	
	# Read in all the data for a specific vehicle
	for k in range(len(all_cars)):
	    file = "data/"+output[k]+"/"+target_car+"_daily_data.txt"
	    threshold = thresholds[k]
	    car = all_cars[k]
	    car.read_in_data(file)
	    car.fill_NA()
	    y_pos, y_neg, y_neu = car.return_sentiment_data()
	    
	    # Other transformations that are available
	    #y_pos,y_neg,y_neu =car.scale_data_Ztransform()
	    #y_pos,y_neg,y_neu =car.scale_data_MinMaxTransform(Xmin=-1,Xmax=1)
	        
	    plt.plot(car.car_data["Dates"],y_pos,"-o",label="T="+str(round(threshold,3)))
	    
	plt.title(target_car,size=20)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.xticks(rotation='vertical')
	plt.xlabel("Dates",size=20)
	plt.ylabel("Positive Sentiment %",size=20)
	plt.savefig("data/"+target_car+"/"+target_car+"_pos_sentiment_curves_all_thresholds.pdf",bbox_inches="tight")
	
	#==========================================================================================================
	
	plt.clf()
	
	# Read in all the data for a specific vehicle
	for k in range(len(all_cars)):
	    file = "data/"+output[k]+"/"+target_car+"_daily_data.txt"
	    threshold = thresholds[k]
	    car = all_cars[k]
	    y_pos, y_neg, y_neu = car.return_sentiment_data()
	    #y_pos,y_neg,y_neu =car.scale_data_Ztransform()
	    #y_pos,y_neg,y_neu =car.scale_data_MinMaxTransform(Xmin=-1,Xmax=1)
	    
	    plt.plot(car.car_data["Dates"],y_neg,"-o",label="T="+str(threshold))
	    
	plt.title(target_car,size=20)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.xticks(rotation='vertical')
	plt.xlabel("Dates",size=20)
	plt.ylabel("Negative Sentiment %",size=20)
	plt.savefig("data/"+target_car+"/"+target_car+"_neg_sentiment_curves_all_thresholds.pdf",bbox_inches="tight")
	
	#===========================================================================================================
	#
	# Let us compute the model weights, W_tp, W_tn, W_fp, W_fn and plot them vs T 
	#
	#===========================================================================================================
	
	file_name="data/sentiment_model_tweet_test_scores.txt"
	T, prob_tp,prob_fp, prob_tn,prob_fn, prob_pos, prob_neg = Vehicle_data.return_model_training_results_prob(file_name)

	
	pos_norm = np.trapz(prob_tp*prob_pos+prob_fp*prob_pos, x=T)
	neg_norm = np.trapz(prob_tn*prob_pos+prob_fn*prob_pos, x=T)
	
	w_tp = prob_tp*prob_pos/pos_norm
	w_fp = prob_fp*prob_pos/pos_norm
	
	w_tn = prob_tn*prob_neg/neg_norm
	w_fn = prob_fn*prob_neg/neg_norm
	
	plt.clf()
	plt.title("Pos signal model weight vs T",size=20)
	plt.plot(T,w_tp,label=r"$W_{tp}(T)$",color="green")
	plt.plot(T,w_fp,label=r"$W_{fp}(T)$",color="green",linestyle=":")
	plt.xticks(size=15)
	plt.yticks(size=15)
	plt.xlabel(r"Threshold $T$",size=20)
	plt.ylabel(r"$W(T)$",size=20)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.savefig("data/"+target_car+"/Model_Weights_Positive.pdf",bbox_inches="tight")
	
	plt.clf()
	plt.title("Neg signal model weight vs T",size=20)
	plt.plot(T,w_tn,label=r"$W_{tn}(T)$",color="red")
	plt.plot(T,w_fn,label=r"$W_{fn}(T)$",color="red",linestyle=":")
	plt.xticks(size=15)
	plt.yticks(size=15)
	plt.xlabel(r"Threshold $T$",size=20)
	plt.ylabel(r"$W(T)$",size=20)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.savefig("data/"+target_car+"/Model_Weights_Negative.pdf",bbox_inches="tight")
	
	#===========================================================================================================
	#===========================================================================================================
	# Now We combine all of the curves
	
	N =len(all_cars[0].car_data["Dates"])
	
	dates = all_cars[0].car_data["Dates"]
	
	y_pos_tot = np.zeros(N)
	y_neg_tot = np.zeros(N)
	y_neu_tot = np.zeros(N)
	
	y_tp = np.zeros(N)
	y_fp = np.zeros(N)
	
	y_tn = np.zeros(N)
	y_fn = np.zeros(N)
	
	# Lopp over all thresholds
	for k in range(len(all_cars)):
		#y_pos,y_neg,y_neu =all_cars[k].return_sentiment_data()
		
		if(sentiment_curve_scaling=='Z-transform'):
			y_pos,y_neg,y_neu =all_cars[k].scale_data_Ztransform()
		elif(sentiment_curve_scaling=='MinMax-transform'):
			y_pos,y_neg,y_neu =all_cars[k].scale_data_MinMaxTransform(Xmin=sentiment_scaling_Xmin,Xmax=sentiment_scaling_Xmax)
		
		#y_pos,y_neg,y_neu =all_cars[k].scale_data_MinMaxTransform(Xmin=-1,Xmax=1)
		#y_pos,y_neg,y_neu =all_cars[k].daily_sentiment_return()
		
		if(remove_outliers==True):
			y_pos = car.remove_outliers(y_pos)
			y_neg = car.remove_outliers(y_neg)
			y_neu = car.remove_outliers(y_neu)
	
		
		#y_pos_tot += prob_pos[k]*prob_tp[k]*y_pos
		y_tp += w_tp[k]*y_pos
		y_fp += w_fn[k]*y_pos
		
		y_tn += w_tn[k]*y_neg
		y_fn += w_fn[k]*y_neg
	    
	
	# Scale the data to improve the sampling
	y_tp = b1_scale*y_tp
	y_fp = b1_scale*y_fp
	
	y_tn = b2_scale*y_tn
	y_fn = b2_scale*y_fn
	
	# Now we plot the normalized Positive and negative sentiment signal curves for both the expected true signals, and the 
	# expected false signal.
	plt.clf() 
	plt.plot(dates,y_tp,label="True Positive signal",color="g")
	plt.plot(dates,y_fp,label="False Positive signal",color="g",alpha=0.4)
	plt.plot([], [], ' ', label="Scale: X_0="+str(b1_scale))
	plt.ylabel("Normalized Positive Signal",size=15)
	plt.xticks(rotation='vertical')
	plt.xlabel("Dates",size=20)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.savefig("data/"+target_car+"/"+target_car+"_normed_pos_signal.pdf",bbox_inches="tight")
	
	plt.clf()
	plt.plot(dates,y_tn,label="True Negative signal",color="r")
	plt.plot(dates,y_fn,label="False Negative signal",color="r",alpha=0.4)
	plt.plot([], [], ' ', label="Scale: X_0="+str(b2_scale))
	plt.ylabel("Normalized Negative Signal",size=15)
	plt.xticks(rotation='vertical')
	plt.xlabel("Dates",size=20)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.savefig("data/"+target_car+"/"+target_car+"_normed_neg_signal.pdf",bbox_inches="tight")
	
	
	# Load in all of the stock Data from appropriate sources
	data_target = pd.read_csv("stock_data/"+target_car+".txt",delimiter=" ")
	data_sp500 = pd.read_csv("stock_data/SandP_500.txt",delimiter=" ")
	
	# Remove the first date so as to match dimensions of the daily returns
	y_tp=y_tp[1:]
	y_tn=y_tn[1:]
	
	# Return the Closing Price of the Target vehicle stock
	y_target_car=np.asarray(data_target["Closing_Price"])
	y_target_car=Vehicle_data().fill_NA_array(y_target_car)
	    
	# Read in the S and P 500 data
	y_sp500=np.asarray(data_sp500["Closing_Price"])
	y_sp500=Vehicle_data().fill_NA_array(y_sp500)
	
	# Convert values to daily returns
	daily_return_target_car=(y_target_car[1:]/y_target_car[0:-1])-1.0
	daily_return_sp500=(y_sp500[1:]/y_sp500[0:-1])-1.0
	
	
	# Remove the outliers in the Data
	if(remove_outliers==True):
		daily_return_target_car = car.remove_outliers(daily_return_target_car)
		daily_return_sp500= car.remove_outliers(daily_return_sp500)
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# Insert code to cheack that stock data has been read in correctly!
	print(y_target_car[1:])
	print(daily_return_target_car)
	print(daily_return_sp500)
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
	# Let us check that the dimensions all agree
	if(len(y_tp) != len(daily_return_target_car)):
		print('TP not the same length as Daily returns')
		print('TP length: ', len(y_tp))
		print('Daily Ret length: ',len(daily_return_target_car),'\n')
		print('Last Date in Daily Ret: ', data_target["Dates"],'\n')
		print('Last Date in S&P500 Ret: ', data_sp500["Dates"],'\n')
		print('Last Date in Sentiment: ', all_cars[0].car_data["Dates"])
		print('Program Exiting')
		exit()
	
	
	# Visualize the results
	plt.clf()
	plt.plot(y_tp,daily_return_target_car,"o",c="g",label="True Positive Sentiment")
	plt.plot(y_tn,daily_return_target_car,"o",c="r",label="True Negative Sentiment")
	plt.plot([], [], ' ', label="Scale: X_0(pos)="+str(b1_scale))
	plt.plot([], [], ' ', label="Scale: X_0(neg)="+str(b2_scale))
	plt.axvline(x=0, color='k')
	plt.axhline(y=0, color='k')
	plt.ylabel("Daily Returns "+target_car,size=20)
	plt.xlabel("Sentiment Signals",size=20)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.savefig("data/"+target_car+"/"+target_car+"_daily_returns_vs_sentiment_data.pdf",bbox_inches="tight")
	
	plt.clf()
	plt.plot(daily_return_sp500,daily_return_target_car,"o")
	plt.axvline(x=0, color='k')
	plt.axhline(y=0, color='k')
	plt.xlabel("Daily Returns S&P500",size=20)
	plt.ylabel("Daily Returns "+target_car,size=20)
	plt.savefig("data/"+target_car+"/"+target_car+"_sp500_vs_daily_returns.pdf",bbox_inches="tight")
	
	#======================================================================================================
	# Here the model is fitted in a bayesian way
	#======================================================================================================	
	x1 = y_tp
	x2 = y_tn
	x3 = daily_return_sp500
	y = daily_return_target_car
	
	plt.clf()
	a_mcmc =[]
	b1_mcmc=[]
	b2_mcmc=[]
	b3_mcmc=[]
	sigma_mcmc=[]
	corr_ytrue_market=[]
	corr_ytrue_pos=[]
	corr_ytrue_neg=[]
	N_vals =[]
	
	# Here we test that the Nmax is not too large compared to available array dimensions
	if(N_conv_max > len(daily_return_sp500)):
		print('Specified Nmax is greater than Dim of Sentiment Array: ')
		print('Changing Upper Bound')
		N_conv_max = len(daily_return_sp500)
	
	
	for N in range(N_conv_min,N_conv_max+1):
		samples_N,a_mcmc_N, b1_mcmc_N,b2_mcmc_N,b3_mcmc_N,sigma_mcmc_N = fit_model(x1,x2,x3,y,N,N_mcmc_walkers,N_mcmc_runs,N_mcmc_burn,target_car,b1_scale,b2_scale)
		corr_ytrue_market_N,corr_ytrue_pos_N,corr_ytrue_neg_N =correlation_analysis(x1,x2,x3,y,samples_N,N,target_car,N_correlation_samples = N_correlation_samples)
		
		N_vals.append(N)
		a_mcmc.append(a_mcmc_N)
		b1_mcmc.append(b1_mcmc_N)
		b2_mcmc.append(b2_mcmc_N)
		b3_mcmc.append(b3_mcmc_N)
		sigma_mcmc.append(sigma_mcmc_N)
		
		corr_ytrue_market.append(corr_ytrue_market_N)
		corr_ytrue_pos.append(corr_ytrue_pos_N)
		corr_ytrue_neg.append(corr_ytrue_neg_N)
	
	alpha_median_convergence = [a_mcmc[k][0] for k in range(len(a_mcmc))]
	
	beta_P_median_convergence = [b1_mcmc[k][0] for k in range(len(b1_mcmc))]
	beta_P_upper_convergence = [b1_mcmc[k][0]+b1_mcmc[k][1] for k in range(len(b1_mcmc))]
	beta_P_lower_convergence = [b1_mcmc[k][0]-b1_mcmc[k][2] for k in range(len(b1_mcmc))]
	
	beta_P_upper96_convergence = [b1_mcmc[k][0]+b1_mcmc[k][3] for k in range(len(b1_mcmc))]
	beta_P_lower96_convergence = [b1_mcmc[k][0]-b1_mcmc[k][4] for k in range(len(b1_mcmc))]
	
	beta_N_median_convergence = [b2_mcmc[k][0] for k in range(len(b2_mcmc))]
	beta_N_upper_convergence = [b2_mcmc[k][0]+b2_mcmc[k][1] for k in range(len(b2_mcmc))]
	beta_N_lower_convergence = [b2_mcmc[k][0]-b2_mcmc[k][2] for k in range(len(b2_mcmc))]
	
	beta_N_upper96_convergence = [b2_mcmc[k][0]+b2_mcmc[k][3] for k in range(len(b2_mcmc))]
	beta_N_lower96_convergence = [b2_mcmc[k][0]-b2_mcmc[k][4] for k in range(len(b2_mcmc))]
	
	beta_M_median_convergence = [b3_mcmc[k][0] for k in range(len(b3_mcmc))]
	beta_M_upper_convergence = [b3_mcmc[k][0]+b3_mcmc[k][1] for k in range(len(b3_mcmc))]
	beta_M_lower_convergence = [b3_mcmc[k][0]-b3_mcmc[k][2] for k in range(len(b3_mcmc))]

	beta_M_upper96_convergence = [b3_mcmc[k][0]+b3_mcmc[k][3] for k in range(len(b3_mcmc))]
	beta_M_lower96_convergence = [b3_mcmc[k][0]-b3_mcmc[k][4] for k in range(len(b3_mcmc))]	
	
	# Correlation vs N
	corr_ytrue_market_median_convergence = [corr_ytrue_market[k][0] for k in range(len(corr_ytrue_market)) ]
	corr_ytrue_market_upper_convergence = [corr_ytrue_market[k][1] for k in range(len(corr_ytrue_market)) ]
	corr_ytrue_market_lower_convergence = [corr_ytrue_market[k][2] for k in range(len(corr_ytrue_market)) ]
	
	corr_ytrue_market_upper96_convergence = [corr_ytrue_market[k][3] for k in range(len(corr_ytrue_market)) ]
	corr_ytrue_market_lower96_convergence = [corr_ytrue_market[k][4] for k in range(len(corr_ytrue_market)) ]

	corr_ytrue_pos_median_convergence = [corr_ytrue_pos[k][0] for k in range(len(corr_ytrue_pos)) ]
	corr_ytrue_pos_upper_convergence = [corr_ytrue_pos[k][1] for k in range(len(corr_ytrue_pos)) ]
	corr_ytrue_pos_lower_convergence = [corr_ytrue_pos[k][2] for k in range(len(corr_ytrue_pos)) ]
	
	corr_ytrue_pos_upper96_convergence = [corr_ytrue_pos[k][3] for k in range(len(corr_ytrue_pos)) ]
	corr_ytrue_pos_lower96_convergence = [corr_ytrue_pos[k][4] for k in range(len(corr_ytrue_pos)) ]


	corr_ytrue_neg_median_convergence = [corr_ytrue_neg[k][0] for k in range(len(corr_ytrue_pos)) ]
	corr_ytrue_neg_upper_convergence = [corr_ytrue_neg[k][1] for k in range(len(corr_ytrue_neg)) ]
	corr_ytrue_neg_lower_convergence = [corr_ytrue_neg[k][2] for k in range(len(corr_ytrue_neg)) ]	
	
	corr_ytrue_neg_upper96_convergence = [corr_ytrue_neg[k][3] for k in range(len(corr_ytrue_neg)) ]
	corr_ytrue_neg_lower96_convergence = [corr_ytrue_neg[k][4] for k in range(len(corr_ytrue_neg)) ]	
	
	
	plt.clf()
	plt.plot(N_vals,beta_P_median_convergence,'-o',color="g")
	plt.plot(N_vals,beta_P_upper_convergence,'--',color="g")
	plt.plot(N_vals,beta_P_lower_convergence,'--',color="g")
	plt.plot(N_vals,beta_P_upper96_convergence,':',color="g")
	plt.plot(N_vals,beta_P_lower96_convergence,':',color="g")
	plt.ylabel(r"$\beta_P$", size=20)
	plt.xlabel(r"$N$", size=20)
	plt.savefig("data/"+target_car+"/"+target_car+"_beta_P_convergence.pdf",bbox_inches="tight")
	
	plt.clf()
	plt.plot(N_vals,corr_ytrue_pos_median_convergence,'-o',color="g")
	plt.plot(N_vals,corr_ytrue_pos_lower_convergence,'--',color="g")
	plt.plot(N_vals,corr_ytrue_pos_upper_convergence,'--',color="g")
	plt.plot(N_vals,corr_ytrue_pos_lower96_convergence,':',color="g")
	plt.plot(N_vals,corr_ytrue_pos_upper96_convergence,':',color="g")
	plt.ylabel(r"$\rho(r,P)$", size=20)
	plt.xlabel(r"$N$", size=20)
	plt.savefig("data/"+target_car+"/"+target_car+"_corr_P_convergence.pdf",bbox_inches="tight")
	
	plt.clf()
	plt.plot(N_vals,beta_N_median_convergence,'-o',color="red")
	plt.plot(N_vals,beta_N_upper_convergence,'--',color="red")
	plt.plot(N_vals,beta_N_lower_convergence,'--',color="red")
	plt.plot(N_vals,beta_N_upper96_convergence,':',color="red")
	plt.plot(N_vals,beta_N_lower96_convergence,':',color="red")
	plt.ylabel(r"$\beta_N$", size=20)
	plt.xlabel(r"$N$", size=20)
	plt.savefig("data/"+target_car+"/"+target_car+"_beta_N_convergence.pdf",bbox_inches="tight")
	
	plt.clf()
	plt.plot(N_vals,corr_ytrue_neg_median_convergence,'-o',color="red")
	plt.plot(N_vals,corr_ytrue_neg_lower_convergence,'--',color="red")
	plt.plot(N_vals,corr_ytrue_neg_upper_convergence,'--',color="red")
	plt.plot(N_vals,corr_ytrue_neg_lower96_convergence,':',color="red")
	plt.plot(N_vals,corr_ytrue_neg_upper96_convergence,':',color="red")
	plt.ylabel(r"$\rho(r,N)$", size=20)
	plt.xlabel(r"$N$", size=20)
	plt.savefig("data/"+target_car+"/"+target_car+"_corr_N_convergence.pdf",bbox_inches="tight")
	
	plt.clf()
	plt.plot(N_vals,beta_M_median_convergence,'-o',color="orange")
	plt.plot(N_vals,beta_M_upper_convergence,'--',color="orange")
	plt.plot(N_vals,beta_M_lower_convergence,'--',color="orange")
	plt.plot(N_vals,beta_M_upper96_convergence,':',color="orange")
	plt.plot(N_vals,beta_M_lower96_convergence,':',color="orange")
	plt.ylabel(r"$\beta_M$", size=20)
	plt.xlabel(r"$N$", size=20)
	plt.savefig("data/"+target_car+"/"+target_car+"_beta_M_convergence.pdf",bbox_inches="tight")
	
	plt.clf()
	plt.plot(N_vals,corr_ytrue_market_median_convergence,'-o',color="orange")
	plt.plot(N_vals,corr_ytrue_market_lower_convergence,'--',color="orange")
	plt.plot(N_vals,corr_ytrue_market_upper_convergence,'--',color="orange")
	plt.plot(N_vals,corr_ytrue_market_lower96_convergence,':',color="orange")
	plt.plot(N_vals,corr_ytrue_market_upper96_convergence,':',color="orange")
	plt.ylabel(r"$\rho(r,M)$", size=20)
	plt.xlabel(r"$N$", size=20)
	plt.savefig("data/"+target_car+"/"+target_car+"_corr_M_convergence.pdf",bbox_inches="tight")
	
	return None
	
	
	
def fit_model(x1,x2,x3,y,N,N_mcmc_walkers,N_mcmc_runs,N_mcmc_burn,target_car,b1_scale,b2_scale):
	'''
	x1 = y_tp
	x2 = y_tn
	x3 = daily_return_sp500
	y = daily_return_target_car
	N: Number of data points used for fitting
	'''
	
	x1 = x1[0:N]
	x2 = x2[0:N]
	x3 = x3[0:N] 
	y = y[0:N]
	
	
    #======================================================================================================
	# Here the model is fitted in a bayesian way
	#======================================================================================================	
	def lnlike(theta,x1,x2,x3,y):
	    
	    alpha,b1,b2,b3,sigma= theta
	    
	    model = alpha+b1*x1+b2*x2 +b3*x3 
	    
	    inv_sigma2 = 1.0/sigma**2
	    
	    s = -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))
	    
	    return s
	
	def lnprior(theta):
	    alpha,b1,b2,b3,sigma = theta
	    
	    if sigma >=1.0e-5 and sigma <= 0.2 and (-100 <= b1 and b1 <= 100) and (-100 <= b2 and b2 <= 100) and (-100 <= b3 and b3 <= 100):
	        return -np.log(sigma)
	    
	    
	    return -np.inf
	
	def lnprob(theta, x1,x2,x3, y):
	    lp = lnprior(theta)
	    if not np.isfinite(lp):
	        return -np.inf
	    return lp + lnlike(theta, x1,x2,x3, y)
	
	
	nll = lambda *args: -lnlike(*args)
	
	result = op.minimize(nll, np.random.rand(5), args=(x1,x2,x3, y))
	result["x"] = np.abs(result["x"])
	
	m_ml, b1_ml,b2_ml,b3_ml,sigma_ml = result["x"]
	
	ndim, nwalkers = 5, N_mcmc_walkers
	pos = [result["x"] + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x1,x2,x3, y))
	sampler.run_mcmc(pos, N_mcmc_runs)
	samples = sampler.chain[:, N_mcmc_burn:, :].reshape((-1, ndim))
	
	N_round = 4

	s=samples[:,0]
	median, q1, q3,q4,q5 = np.percentile(s, 50), np.percentile(s, 16), np.percentile(s, 84), np.percentile(s,2),np.percentile(s,98)
	plt.clf()
	plt.axvline(x=median, color='k')
	plt.axvline(x=q1, color='k',linestyle="--",label="68% Confidence interval")
	plt.axvline(x=q3, color='k',linestyle="--")
	plt.plot([], [], ' ', label=str(round(median,N_round))+" "+"^{+"+str(round(q3-median,N_round))+"}"+"_{-"+str(round(median-q1,N_round))+"}")
	plt.axvline(x=q4, color='k',linestyle=":",label="96% Confidence interval")
	plt.axvline(x=q5, color='k',linestyle=":")
	plt.plot([], [], ' ', label=str(round(median,N_round))+" "+"^{+"+str(round(q5-median,N_round))+"}"+"_{-"+str(round(median-q4,N_round))+"}")
	plt.ylabel(r"$P(\alpha )$", size=20)
	plt.xlabel(r"$\alpha$", size=20)
	plt.hist(s,bins=50,density=True,alpha=0.4,color="b")
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.title(target_car,size=20)
	plt.savefig("data/"+target_car+"/"+target_car+"_alpha_pdf="+str(N)+".pdf",bbox_inches="tight")
	
	s=samples[:,1]
	#s=s*b1_scale
	median, q1, q3,q4,q5 = np.percentile(s, 50), np.percentile(s, 16), np.percentile(s, 84), np.percentile(s,2),np.percentile(s,98)
	plt.clf()
	plt.plot([], [], ' ', label="Scale: X_0="+str(b1_scale))
	plt.axvline(x=median, color='k')
	plt.axvline(x=q1, color='k',linestyle="--",label="68% Confidence interval")
	plt.axvline(x=q3, color='k',linestyle="--")
	plt.plot([], [], ' ', label=str(round(median,N_round))+" "+"^{+"+str(round(q3-median,N_round))+"}"+"_{-"+str(round(median-q1,N_round))+"}")
	plt.axvline(x=q4, color='k',linestyle=":",label="96% Confidence interval")
	plt.axvline(x=q5, color='k',linestyle=":")
	plt.plot([], [], ' ', label=str(round(median,N_round))+" "+"^{+"+str(round(q5-median,N_round))+"}"+"_{-"+str(round(median-q4,N_round))+"}")
	plt.ylabel(r"$P(\beta_P )$", size=20)
	plt.xlabel(r"$\beta_P/X_0$", size=20)
	plt.hist(s,bins=50,density=True,alpha=0.6,color="g")
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.title(target_car,size=20)
	plt.savefig("data/"+target_car+"/"+target_car+"_beta_P_pdf="+str(N)+".pdf",bbox_inches="tight")
	
	s=samples[:,2]
	#s=s*b2_scale
	median, q1, q3,q4,q5 = np.percentile(s, 50), np.percentile(s, 16), np.percentile(s, 84), np.percentile(s,2),np.percentile(s,98)
	plt.clf()
	plt.plot([], [], ' ', label="Scale: X_0="+str(b2_scale))
	plt.axvline(x=median, color='k')
	plt.axvline(x=q1, color='k',linestyle="--",label="68% Confidence interval")
	plt.axvline(x=q3, color='k',linestyle="--")
	plt.plot([], [], ' ', label=str(round(median,N_round))+" "+"^{+"+str(round(q3-median,N_round))+"}"+"_{-"+str(round(median-q1,N_round))+"}")
	plt.axvline(x=q4, color='k',linestyle=":")
	plt.axvline(x=q5, color='k',linestyle=":",label="96% Confidence interval")
	plt.plot([], [], ' ', label=str(round(median,N_round))+" "+"^{+"+str(round(q5-median,N_round))+"}"+"_{-"+str(round(median-q4,N_round))+"}")
	plt.ylabel(r"$P(\beta_N )$", size=20)
	plt.xlabel(r"$\beta_N/X_0$", size=20)
	plt.hist(s,bins=50,density=True,alpha=0.6,color="red")
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.title(target_car,size=20)
	plt.savefig("data/"+target_car+"/"+target_car+"_beta_N_pdf="+str(N)+".pdf",bbox_inches="tight")
	
	s=samples[:,3]
	median, q1, q3,q4,q5 = np.percentile(s, 50), np.percentile(s, 16), np.percentile(s, 84), np.percentile(s,2),np.percentile(s,98)
	plt.clf()
	plt.axvline(x=median, color='k')
	plt.axvline(x=q1, color='k',linestyle="--",label="68% Confidence interval")
	plt.axvline(x=q3, color='k',linestyle="--")
	plt.plot([], [], ' ', label=str(round(median,N_round))+" "+"^{+"+str(round(q3-median,N_round))+"}"+"_{-"+str(round(median-q1,N_round))+"}")
	plt.axvline(x=q4, color='k',linestyle=":")
	plt.axvline(x=q5, color='k',linestyle=":",label="96% Confidence interval")
	plt.plot([], [], ' ', label=str(round(median,N_round))+" "+"^{+"+str(round(q5-median,N_round))+"}"+"_{-"+str(round(median-q4,N_round))+"}")
	plt.ylabel(r"$P(\beta_M)$", size=20)
	plt.xlabel(r"$\beta_M$", size=20)
	plt.hist(s,bins=50,density=True,alpha=1.0,color="orange")
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.title(target_car,size=20)
	plt.savefig("data/"+target_car+"/"+target_car+"_beta_M_pdf="+str(N)+".pdf",bbox_inches="tight")
	
	#fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
	s=samples[:,4]
	median, q1, q3,q4,q5 = np.percentile(s, 50), np.percentile(s, 16), np.percentile(s, 84), np.percentile(s,2),np.percentile(s,98)
	plt.clf()
	plt.axvline(x=median, color='k')
	plt.axvline(x=q1, color='k',linestyle="--",label="68% Confidence interval")
	plt.axvline(x=q3, color='k',linestyle="--")
	plt.plot([], [], ' ', label=str(round(median,N_round))+" "+"^{+"+str(round(q3-median,N_round))+"}"+"_{-"+str(round(median-q1,N_round))+"}")
	plt.axvline(x=q4, color='k',linestyle=":")
	plt.axvline(x=q5, color='k',linestyle=":",label="96% Confidence interval")
	plt.plot([], [], ' ', label=str(round(median,N_round))+" "+"^{+"+str(round(q5-median,N_round))+"}"+"_{-"+str(round(median-q4,N_round))+"}")
	plt.ylabel(r"$P(\sigma)$", size=20)
	plt.xlabel(r"$\sigma$", size=20)
	plt.hist(s,bins=50,density=True,alpha=0.4,color="blue")
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.title(target_car,size=20)
	plt.savefig("data/"+target_car+"/"+target_car+"_sigma_pdf="+str(N)+".pdf",bbox_inches="tight")
	
	
	#samples[:,1]=samples[:,1]*b1_scale
	#samples[:,2]=samples[:,2]*b2_scale
	
	# This is used to generate the corner plot
	#fig = corner.corner(samples,labels=["$a$", "$B_p$","$B_n$","$B_M$","$\sigma$"], quantiles=[0.16, 0.5, 0.84],
	#                    show_titles=True,
	#                    title_kwargs={"fontsize": 12})
	#fig.savefig("data/"+target_car+"/"+target_car+"_corner_plot.pdf")
	
	
	a_mcmc, b1_mcmc,b2_mcmc,b3_mcmc,sigma_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0],v[4]-v[1],v[1]-v[3]),
                             zip(*np.percentile(samples, [16, 50, 84,2,98],
                                                axis=0)))

	print("====================================================\n")
	print("Parameter estimation for: "+ target_car+"\n")
	print("The number of Data Points is : ", N,'\n')
	
	print("The 68% Confidence regions")
	print("a_mcmc = %.04f ^{+%0.03f}_{-%0.03f}"%(a_mcmc[0],a_mcmc[1],a_mcmc[2]))
	print("bP_mcmc = %.04f ^{+%0.03f}_{-%0.03f}"%(b1_mcmc[0],b1_mcmc[1],b1_mcmc[2]))
	print("bN_mcmc = %.04f ^{+%0.03f}_{-%0.03f}"%(b2_mcmc[0],b2_mcmc[1],b2_mcmc[2]))
	print("bM_mcmc = %.04f ^{+%0.03f}_{-%0.03f}"%(b3_mcmc[0],b3_mcmc[1],b3_mcmc[2]))
	print("sigma_mcmc = %.04f ^{+%0.03f}_{-%0.03f}"%(sigma_mcmc[0],sigma_mcmc[1],sigma_mcmc[2]))
	
	print("\n")
	print("The 96% Confidence regions")
	print("a_mcmc = %.04f ^{+%0.03f}_{-%0.03f}"%(a_mcmc[0],a_mcmc[3],a_mcmc[4]))
	print("bP_mcmc = %.04f ^{+%0.03f}_{-%0.03f}"%(b1_mcmc[0],b1_mcmc[3],b1_mcmc[4]))
	print("bN_mcmc = %.04f ^{+%0.03f}_{-%0.03f}"%(b2_mcmc[0],b2_mcmc[3],b2_mcmc[4]))
	print("bM_mcmc = %.04f ^{+%0.03f}_{-%0.03f}"%(b3_mcmc[0],b3_mcmc[3],b3_mcmc[4]))
	print("sigma_mcmc = %.04f ^{+%0.03f}_{-%0.03f}"%(sigma_mcmc[0],sigma_mcmc[3],sigma_mcmc[4]),"\n")
	
	print("Correlation between B1 and B2:", np.corrcoef(samples[:,1], (samples[:,2]))[0,1])
	print("====================================================\n")
	
	#=============================================================================================
	# Residual plots
	#=============================================================================================
	
	model = a_mcmc[0]+b1_mcmc[0]*x1+b2_mcmc[0]*x2 +b3_mcmc[0]*x3     
	res = (y-model)
	
	print("residual mean: %.04f"%res.mean())
	print("residual std: %.04f"%res.std())
	
	plt.clf()
	plt.plot(res,"o")
	plt.title("Mean: "+str(np.round(res.mean(),3))+" +/- "+str(np.round(res.std(),4)) ,size=20)
	plt.ylabel("Residuals",size=20)
	plt.xlabel("Data Point", size=20)
	plt.axhline(y=0, color='k')
	plt.savefig("data/"+target_car+"/"+target_car+"_model_residuals_N="+str(N)+".pdf",bbox_inches="tight")
	#======================================================================================================
	
	return samples,a_mcmc, b1_mcmc,b2_mcmc,b3_mcmc,sigma_mcmc





def correlation_analysis(x1,x2,x3,y,samples,N,target_car,N_round=4,N_correlation_samples=50000):
	'''
	x1 = y_tp
	x2 = y_tn
	x3 = daily_return_sp500
	y = daily_return_target_car
	samples: Samples generated from mcmc
	N: The number of data points to use for the fits
	target_car: The target vehicle names
	N_round: How many digits to round to
	N_correlation_samples: Number of samples to use to generate posteriors of correlation coeff.
	'''
	x1 = x1[0:N]
	x2 = x2[0:N]
	x3 = x3[0:N] 
	y = y[0:N]
	
	# Now we compute the distribution of Correlation coefficients 
	alpha_vec =samples[:,0]
	betaP_vec =samples[:,1]
	betaN_vec =samples[:,2]
	betaM_vec =samples[:,3]
	
	
	def model(alpha,betaM,betaP,betaN,x1,x2,x3):
	    '''
	    x1: Positive signal
	    x2: Negative signal
	    x3: Sp500
	    '''
	    
	    r_pred = alpha+betaP*x1+betaN*x2+betaM*x3
	    
	    return r_pred
	
	def pearson_corr(alpha,betaM,betaP,betaN,x1,x2,x3,y_true):
	    '''
	    Pearsons Correlation coefficient
	    '''
	    
	    y_prediction = model(alpha,betaM,betaP,betaN,x1,x2,x3)
	    
	    pos_residual = (y_true-alpha-betaN*x2-betaM*x3)
	    neg_residual = (y_true-alpha-betaP*x1-betaM*x3)
	    market_residual = (y_true-alpha-betaN*x2-betaP*x1)
	    
	    rho_pos =np.corrcoef(x1, pos_residual)
	    rho_neg = np.corrcoef(x2, neg_residual)
	    rho_market =np.corrcoef(x3, market_residual)
	    rho_overall = np.corrcoef(y_prediction, y_true)
	    
	    return rho_pos[0][1],rho_neg[0][1],rho_market[0][1],rho_overall[0][1]
	
	
	pos_corr =[]
	neg_corr =[]
	market_corr =[]
	overall_corr =[]
	
	for k in range(N_correlation_samples):
	    alpha,betaM,betaP,betaN = alpha_vec[k],betaM_vec[k],betaP_vec[k],betaN_vec[k]
	    pos_corr.append(pearson_corr(alpha,betaM,betaP,betaN,x1,x2,x3,y)[0])
	    neg_corr.append(pearson_corr(alpha,betaM,betaP,betaN,x1,x2,x3,y)[1])
	    market_corr.append(pearson_corr(alpha,betaM,betaP,betaN,x1,x2,x3,y)[2])
	    overall_corr.append(pearson_corr(alpha,betaM,betaP,betaN,x1,x2,x3,y)[3])
	
	    
	print('Corr(y_true,Pos): ',np.corrcoef(y,x1)[0][1])
	s = pos_corr
	median, q1, q3,q4,q5 = np.percentile(s, 50), np.percentile(s, 16), np.percentile(s, 84), np.percentile(s,2),np.percentile(s,98)
	corr_ytrue_pos = np.asarray([median,q1,q3,q4,q5])
	plt.clf()
	plt.axvline(x=median, color='k')
	plt.axvline(x=q1, color='k',linestyle="--",label="68% Confidence interval")
	plt.axvline(x=q3, color='k',linestyle="--")
	plt.plot([], [], ' ', label=str(round(median,N_round))+" "+"^{+"+str(round(q3-median,N_round))+"}"+"_{-"+str(round(median-q1,N_round))+"}")
	plt.axvline(x=q4, color='k',linestyle=":")
	plt.axvline(x=q5, color='k',linestyle=":",label="96% Confidence interval")
	plt.plot([], [], ' ', label=str(round(median,N_round))+" "+"^{+"+str(round(q5-median,N_round))+"}"+"_{-"+str(round(median-q4,N_round))+"}")
	plt.ylabel(r"$P(\rho(P,r))$", size=20)
	plt.xlabel(r"$\rho(P,r)$", size=20)
	plt.hist(s,bins=50,density=True,alpha=0.6,color="Green")
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.title(target_car+ ' Pos Sentiment Corr. Coeff.',size=20)
	plt.savefig("data/"+target_car+"/"+target_car+"_Pos_rho_pdf="+str(N)+".pdf",bbox_inches="tight")
	
	print('Corr(y_True,Neg): ',np.corrcoef(y,x2)[0][1])
	s = neg_corr
	median, q1, q3,q4,q5 = np.percentile(s, 50), np.percentile(s, 16), np.percentile(s, 84), np.percentile(s,2),np.percentile(s,98)
	corr_ytrue_neg = np.asarray([median,q1,q3,q4,q5])
	plt.clf()
	plt.axvline(x=median, color='k')
	plt.axvline(x=q1, color='k',linestyle="--",label="68% Confidence interval")
	plt.axvline(x=q3, color='k',linestyle="--")
	plt.plot([], [], ' ', label=str(round(median,N_round))+" "+"^{+"+str(round(q3-median,N_round))+"}"+"_{-"+str(round(median-q1,N_round))+"}")
	plt.axvline(x=q4, color='k',linestyle=":")
	plt.axvline(x=q5, color='k',linestyle=":",label="96% Confidence interval")
	plt.plot([], [], ' ', label=str(round(median,N_round))+" "+"^{+"+str(round(q5-median,N_round))+"}"+"_{-"+str(round(median-q4,N_round))+"}")
	plt.ylabel(r"$P(\rho(N,r))$", size=20)
	plt.xlabel(r"$\rho(N,r)$", size=20)
	plt.hist(s,bins=50,density=True,alpha=0.6,color="Red")
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.title(target_car+ ' Neg Sentiment Corr. Coeff.',size=20)
	plt.savefig("data/"+target_car+"/"+target_car+"_Neg_rho_pdf="+str(N)+".pdf",bbox_inches="tight")
	
	print('Corr(y_true,Market): ',np.corrcoef(y,x3)[0][1])
	s = market_corr
	median, q1, q3,q4,q5 = np.percentile(s, 50), np.percentile(s, 16), np.percentile(s, 84), np.percentile(s,2),np.percentile(s,98)
	corr_ytrue_market = np.asarray([median,q1,q3,q4,q5])
	plt.clf()
	plt.axvline(x=median, color='k')
	plt.axvline(x=q1, color='k',linestyle="--",label="68% Confidence interval")
	plt.axvline(x=q3, color='k',linestyle="--")
	plt.plot([], [], ' ', label=str(round(median,N_round))+" "+"^{+"+str(round(q3-median,N_round))+"}"+"_{-"+str(round(median-q1,N_round))+"}")
	plt.axvline(x=q4, color='k',linestyle=":")
	plt.axvline(x=q5, color='k',linestyle=":",label="96% Confidence interval")
	plt.plot([], [], ' ', label=str(round(median,N_round))+" "+"^{+"+str(round(q5-median,N_round))+"}"+"_{-"+str(round(median-q4,N_round))+"}")
	plt.ylabel(r"$P(\rho(M,r))$", size=20)
	plt.xlabel(r"$\rho(M,r)$", size=20)
	plt.hist(s,bins=50,density=True,alpha=0.8,color="Orange")
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.title(target_car+ ' Market Corr. Coeff.',size=20)
	plt.savefig("data/"+target_car+"/"+target_car+"_Market_rho_pdf="+str(N)+".pdf",bbox_inches="tight")
	
	
	return corr_ytrue_market,corr_ytrue_pos,corr_ytrue_neg
	
