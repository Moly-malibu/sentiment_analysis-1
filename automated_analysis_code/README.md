# Automated Analysis

This folder will go through all the steps required from training the model,
to generating the time series of the sentiment, to carrying out the Bayesian analysis 
of the correlation coefficients. 

```
{
    "Model_Training": false,
    "Model_Training_Split": 0.8,
    "Model_Review_Sizes": 1000,
    "Model_Threshold_Values": {
		"Tmin": 0.0,
		"Tmax": 1.1,
		"dT": 0.1
		},
	"Five_Star_Scale_Review_Training_Data": ["training_data/reviews_Automotive_5.json",
	"training_data/reviews_Office_Products_5.json"], 
    "SQL_Database": "databases/keyword_based_database_2019.db",
    "Car_Companies": ["Tesla"],
    "Time_Stamps": ["Jan 14", "%Jan 14 %","%Jan 15 %","%Jan 16 %","%Jan 17 %","%Jan 18 %","%Jan 19 %","%Jan 20 %",
    "%Jan 21 %", "%Jan 22 %", "%Jan 23 %","%Jan 24 %", "%Jan 25 %", "%Jan 26 %",
    "%Jan 27 %", "%Jan 28 %", "%Jan 29 %","%Jan 30 %", "%Jan 31 %", "%Feb 01 %",
    "%Feb 02 %", "%Feb 03 %", "%Feb 04 %","%Feb 05 %", "%Feb 06 %", "%Feb 07 %",
    "%Feb 08 %", "%Feb 09 %", "%Feb 10 %","%Feb 11 %", "%Feb 12 %", "%Feb 13 %",
    "%Feb 14 %", "%Feb 15 %", "%Feb 16 %","%Feb 17 %", "%Feb 18 %", "%Feb 19 %"],
    "Compute_Model_Metrics": false,
    "Daily_Time_Series_Analysis": false,
    "Bayesian_Correlation_Analysis": true,
    "Remove_outliers": true,
    "Sentiment_Curve_Scaling": {
		"Transform":"MinMax-transform",
		"Xmin":0.0,
		"Xmax":1.0
	},
    "X0_pos_scale": 1.0e-3,
    "X0_neg_scale": 1.0e-3,
    "N_mcmc_runs":2000,
    "N_mcmc_walkers":20, 
    "N_mcmc_burn":100,
    "N_correlation_samples": 6000,
    "Convergence_Plot": {
		"Nmin": 5,
		"Nmax": 36
		}
}
```
