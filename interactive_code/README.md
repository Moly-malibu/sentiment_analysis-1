# Full Sentiment Model


This repository contains all programs that are used for the twitter sentiment analysis
applied to automotive tweets. 

# 1_sentiment_analysis_model
* This Jupyter notebook will read in the reviews from the data sources  
* Once read in, program will process text to filter out punctuation, stop words, lemmantize  
* Using the processed documents, the program will count the frequency of the words in the documents and 
create a vocabulary of the most frequently used words.
* Once the vocabulary is built, all reviews are converted into feature vectors based on the vocabulary
* The data set is split into test vs training, models are trained then tested to see which are the most accurate
* trained Logistic Regression and Naive Bayes classifiers are exported
* Vocabulary: **articles-raw.pkl**
* Naive Bayes Model: **lr_sentiment_model.sav**
* Logistic Regression Model: **nb_sentiment_model.sav**

# 2_topic_modelling
* 


# 3_final_analysis
* This Jupyter notebook will read in, process all tweets and compute the sentiment of all car brands using the SQL queries and models that
were trained previously.
* This is carried out over all elements in the database

# 4_final_analysis_time_series
* This jupyter notebook computes the sentiments for all car brands for a set of specific days.
* A time series scatter-plot is generated based on the data

# 5_time_series_comparisons
* This jupyter notebook generates plots of all car brands 


# Data Sources:
* Amazon Reviews 
* Yelp Reviews
