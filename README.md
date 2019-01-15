# Sentiment Analysis of Automotive Companies
 
This repository carries out sentiment analysis on website review data from different sources. Please look at the main [Jupyter notebook](Review_Analysis.ipynb) 


This code was written for a sentiment analysis model for [__APIthinking__](https://www.apithinking.de/en/).

<img src="ApiThinking_RGB_black.png"
     alt="ApiThinking icon"
     style="float: left; margin-right: 10px;" />


# Requirements

To export environment file

activate <environment-name>
```
$conda env export > <environment-name>.yml
```

For other person to use the environment
```
$conda env create -f <environment-name>.yml
```

## Contents

### 1. [Sentiment analysis model](sentiment_model/)  
This model was trained using a combination of Amazon customer review data along with Yelp review data.
The final chosen model was a Naive Bayes Classifier.


### 2. [Twitter Data Collector Code](server_code/)
This folder contains all of the code required to run the Twitter data listener and collect 
the twitter data as a SQL database.
