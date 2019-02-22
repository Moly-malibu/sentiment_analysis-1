# Project Summary  
Is the sentiment of twitter data about specific car brands, related to the stock market returns of the brand? 

This repository contains all of the code required to set-up the twitter-listener, collect data and construct a machine-learning model that analyses the daily sentiment of tweets related to automotive companies. Finally, once the time series of the daily sentiments are calculated using a Bayesian-weighted average, a Bayesian linear regression model is fitted relating the calculated sentiment time-series aswell as the daily market return to the daily return of the automotive stock.

Using the Bayesian linear regression, it is then possible to generate posteriors for the model parameters with qunaitfied uncertainties, and finally place bounds on the correlations between twitter sentiments and the daily returns of the stock.    


# Sentiment Analysis of Automotive Companies
 
This repository carries out sentiment analysis on website review data from different sources. Please look at the main [Jupyter notebook](Review_Analysis.ipynb) 


This code was written for a sentiment analysis model for [__ApiThinking__](https://www.apithinking.de/en/).

<img src="ApiThinking_RGB_black.png"
     href = "https://www.apithinking.de/en/"
     alt="ApiThinking icon"
     style="float: left; margin-right: 10px;" />


# Setup instructions

### 1. Install conda
Choose the version of Anaconda that you wish to download, for example
```
$  wget https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh  
$ bash Anaconda3-2018.12-Linux-x86_64.sh
```

### 2. Create Conda enviroment
After conda has been installed. The enviroment to run the code needs to be created. The enviroment that is used
for this program is found in __py35.yml__. To create this enviroment:  

```
$ conda env create -f py35.yml
```

### 3. Run conda enviroment
To run the conda envioment that was created,

```
$ source activate py35
```

It may be necessary to download Spacys word embedding model,  

```
python -m spacy download en_core_web_lg
```

### 4. Export changes to conda enviroment  
If you have made changes to the enviroment needed to run the code in the repository, export the enviroment using,  
```
$conda env export > <environment-name>.yml
```

# Repository contents

### 1. [Interactive Sentiment analysis model](interactive_code/)  
This folder contains all steps in the project from training the model using a combination of Amazon customer review data along with Yelp review data. The final chosen model was voting ensemble of a Naive Bayes Classifier, Logistic regression and the TextBlob prebuilt classifier.

### 2. [Automated Sentiment Analysis code](automated_analysis_code/)  
This folder contains the code that will carry out all steps in the analysis automatically, according to the specified input parameters.

### 3. [Twitter data collection code](server_code/)
This folder contains all of the code required to run the Twitter data listener and collect 
the twitter data as a SQL database.
