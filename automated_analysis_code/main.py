from source.read_input import  *
from source.model_training import model_training
from source.generate_time_series import generate_sentiment_time_series
from source.bayesian_weight_statistics import compute_model_statistics
from source.bayesian_correlation_analysis import compute_correlation_posteriors
import json
from pprint import pprint
import numpy as np
import sys

if __name__ == "__main__":
	
	# Read the input file
	input_file = str(sys.argv[1])
	
	with open(input_file) as f:
		data = json.load(f)
	
	
	pprint(data)
	
	
	if(data["Model_Training"]==True):
		print("Model will be trained!")
		model_training(data["Five_Star_Scale_Review_Training_Data"])
	
	# After model has been trained, we can generate the time series data
	Tmin = data["Model_Threshold_Values"]["Tmin"]
	Tmax = data["Model_Threshold_Values"]["Tmax"]
	dT = data["Model_Threshold_Values"]["dT"]
	thresholds = np.arange(Tmin,Tmax,dT)
	
	time_stamps = data["Time_Stamps"]
	car_companies = data["Car_Companies"]
	database = data["SQL_Database"]
	
	# Generate the statistics of the model for Bayesian weights
	if(data["Compute_Model_Metrics"]== True):
		compute_model_statistics(thresholds)
	
	# This generates the time series data for all car brands if specified
	if(data["Daily_Time_Series_Analysis"]== True):
		generate_sentiment_time_series(time_stamps,car_companies,thresholds,database)
	
	# Carry out the Bayesian model analysis with convergence plots
	
	N_conv_min = data["Convergence_Plot"]["Nmin"]
	N_conv_max = data["Convergence_Plot"]["Nmax"]
	N_mcmc_burn = data["N_mcmc_burn"]
	N_mcmc_walkers=data["N_mcmc_walkers"]
	N_mcmc_runs =data["N_mcmc_runs"]
	N_correlation_samples = data["N_correlation_samples"]
	
	
	for target_car in car_companies:
		compute_correlation_posteriors(target_car,thresholds,N_conv_min,N_conv_max,N_mcmc_burn,N_mcmc_walkers,N_mcmc_runs,N_correlation_samples)
	
