'''
This Code will train the Sentiment analysis model
'''


import warnings
warnings.filterwarnings('ignore')

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from source.process_text import clean_up_text 
from sklearn.feature_extraction.text import TfidfVectorizer
import operator
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from numpy.random import randint
from textblob import TextBlob

from sklearn.externals import joblib

	#-------------------------------------------------------------------
	# These are the parameters of the model we will train
	#-------------------------------------------------------------------
	#n_reviews = 70000 # The number of reviews to use from each data sets
	#train_size = 0.8 # Splits data into two groups which will be further divided
	#random_state = 10 # For reproducibility 
	#-------------------------------------------------------------------


def model_training(amazon_review_data, n_reviews=1000,twitter_database_sentiment_reviews=70000,random_state=1,train_size=0.8):
	'''
	This Program will
	'''
	
	# Read in the Yelp review data
	yelp = pd.read_csv('training_data/yelp.csv')
	yelp = yelp.sample(frac=1)
	
	
	columns = ['review','rating']
	data_amazon = pd.DataFrame(columns=columns)
	
	
	
	for data_set in amazon_review_data:
		
		data_frame =  pd.read_json(str(data_set), lines=True)
		data_frame = data_frame.sample(frac=1) # Shuffle the data frame
		
		# Append the all of the Amazon review data into a single file
		data_amazon = data_amazon.append(data_frame,ignore_index=True)
		
	
	#=====================================================================================================
	# Here we read in the twitter sentiment data and map it to our prefered labels
	#=====================================================================================================
	
	def func(true_sentiment):
	    
	    y_true = None
	    
	    if(true_sentiment==0):
	        y_true = 1
	    elif(true_sentiment==4):
	        y_true = 5
	    
	    return y_true
	
	# Specify the name of the file, along with the name of the columns
	cols = ['sentiment','id','date','query_string','user','text']
	
	# Read in the data, randomly shuffle and reset the index
	data_twitter=pd.read_csv('training_data/training.1600000.processed.noemoticon.csv',header=None, names=cols,encoding = "ISO-8859-1")
	data_twitter=data_twitter.sample(frac=1)
	data_twitter=data_twitter.reset_index(drop=True)
	
	# Take a subset of the data for faster evaluation
	data_twitter = data_twitter[0:twitter_database_sentiment_reviews]
	data_twitter["text"]=data_twitter["text"].apply(func)
	
	#===============================================================================================================
	# Create a new dataframe which will be used to aggregate all reviews
	columns = ['review','rating']
	df = pd.DataFrame(columns=columns)
	
	
	print('---------------------------------------------')
	print('Total Yelp reviews: ',len(yelp))
	print('Total Amazon reviews: ',len(data_amazon))
	print("Total Twitter Data reviews: ", len(data_twitter))
	print('---------------------------------------------')
	
	# Here we read in the reviews from all of the data sets and aggregate them into one data frame
	for i in tqdm(range(0,min(n_reviews,len(yelp)))):
	    df=df.append({"review":yelp["text"][i],"rating":yelp["stars"][i]},ignore_index=True)
	    
	for i in tqdm(range(0,min(n_reviews,len(data_twitter)))):
	    df=df.append({"review":data_twitter["text"][i],"rating":data_twitter["sentiment"][i]},ignore_index=True)
	
	# Load in all of the Amazon Reviews
	for i in tqdm(range(0,min(n_reviews,len(data_amazon)))):
	    df=df.append({"review":data_amazon['reviewText'][i],"rating": int(data_amazon["overall"][i])},ignore_index=True)

	# Find the specific star rated reviews
	df_1_star = df[df["rating"]== 1]
	df_2_star = df[df["rating"]== 2]
	df_3_star = df[df["rating"]== 3]
	df_4_star = df[df["rating"]== 4]
	df_5_star = df[df["rating"]== 5]
	
	print("")
	print('Raw reviews: ')
	print("1-Star reviews:",len(df_1_star))
	print("2-Star reviews:",len(df_2_star))
	print("3-Star reviews:",len(df_3_star))
	print("4-Star reviews:",len(df_4_star))
	print("5-Star reviews:",len(df_5_star))
	print('\n')
	
	# Aggregate all of the review data
	min_stars= min(df_1_star.count()[0],df_5_star.count()[0])
	
	
	# Randomize the data set
	df = df.sample(frac=1)
	
	df_1_star = df[df["rating"]== 1].head(min_stars)
	df_2_star = df[df["rating"]== 2].head(min_stars)
	df_3_star = df[df["rating"]== 3].head(min_stars)
	df_4_star = df[df["rating"]== 4].head(min_stars)
	df_5_star = df[df["rating"]== 5].head(min_stars)
	
	print("")
	print('Balaced reviews: ')
	print("1-Star reviews:",len(df_1_star))
	print("2-Star reviews:",len(df_2_star))
	print("3-Star reviews:",len(df_3_star))
	print("4-Star reviews:",len(df_4_star))
	print("5-Star reviews:",len(df_5_star))
	print('\n')
	
	# Combine all of the dataframes
	frames = [df_1_star,df_2_star,df_3_star,df_4_star,df_5_star]
	df_balanced = pd.concat(frames)
	
	# Shuffle the data frame to randomize everything
	df_balanced = df_balanced.sample(frac=1)
	df_balanced.index = range(0,df_balanced.shape[0]) # Relabel the indices
	df_balanced.head()
	
	print('The total number of reviews: ', len(df_balanced))	
	
	#---------------------------------------------------------------------------------------
	# Now the Text is processed 
	# Let us now generate a subset of the data to create a vocabulary with
	n_sample = len(df_balanced)
	dict_sample = []
	
	# Now we preprocess the entire balanced data set
	for i in tqdm(range(0,n_sample)):
	    sentence = str(df_balanced["review"][i])
	    dict_sample.append(clean_up_text(sentence))
	
	# Convert to numpy array
	dict_sample = np.asarray(dict_sample)
	
	#-------------------------------------------------------------------
	# Use TDIF vectorizer to create the vocabulary
	#-------------------------------------------------------------------
	custom_stop_words = []
	
	with open( "source/stopwords.txt", "r" ) as fin:
	    for line in fin.readlines():
	        custom_stop_words.append( line.strip() )
	
	#----------------------------------------------------------------------------------------------
	# Here we vectorize the dictionary sample
	vectorizer = TfidfVectorizer(stop_words = custom_stop_words,min_df = 20)
	A = vectorizer.fit_transform(dict_sample)
	print( "Created %d X %d TF-IDF-normalized document-term matrix" % (A.shape[0], A.shape[1]) )
	print("")
	#----------------------------------------------------------------------------------------------
	
	# extract the resulting vocabulary
	terms = vectorizer.get_feature_names()
	
	def rank_terms( A, terms ):
	    # get the sums over each column
	    sums = A.sum(axis=0)
	    # map weights to the terms
	    weights = {}
	    for col, term in enumerate(terms):
	        weights[term] = sums[0,col]
	        
	    # rank the terms by their weight over all documents
	    return sorted(weights.items(), key=operator.itemgetter(1), reverse=True)
	
	ranking = rank_terms( A, terms )
	for i, pair in enumerate( ranking[0:20] ):
	    print( "%02d. %s (%.2f)" % ( i+1, pair[0], pair[1] ) )
	    
	# Write the vocabulary to a file
	f = open("data/vocabulary.txt", "w")
	for i, pair in enumerate( ranking):
	    f.write( "%02d. %s (%.2f) \n" % ( i+1, pair[0], pair[1] ) )
	f.close()
	#---------------------------------------------------------------------------
	#
	# Now the Data is collected into a feature matrix to be used for training
	#
	#----------------------------------------------------------------------------
	# Now we must transform our original review data into a feature Matrix
	y = df_balanced["rating"]
	
	# Now we must transform our original review data into a feature Matrix
	
	X = []
	y = []
	
	# converting the data frame into a feature matrix
	for i in tqdm(range(0,df_balanced.shape[0])):
	    
	    r = df_balanced["rating"][i]
	    
	    if(r==1 or r==2):
	        yi =-1
	        X.append(df_balanced["review"][i])
	        y.append(yi)
	    elif(r==5 or r==4):
	        yi=1
	        X.append(df_balanced["review"][i])
	        y.append(yi)
	print('Balanced Dataset: ',len(y))        
	
	## We store the remaining data 
	df_final = df_balanced.drop(df_balanced[(df_balanced.rating == 2) | (df_balanced.rating == 3)| (df_balanced.rating== 4)].index)
	
	
	X = np.asarray(X)
	y = np.asarray(y)
	
	# Converts the Document matrix consisting of strings into arrays according to the dictionary that we built previously
	X= vectorizer.transform(X)
	
	print('X- Feature matrix: ', X.shape)
	
	# Now we split the data into a training and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)
	
	print('X_train: ', len(y_train))
	print('X_test: ', len(y_test))
	
	
	#---------------------------------------------------------------------------
	# Different models are then trained
	#---------------------------------------------------------------------------
	
	# Now we train different models
	nb = MultinomialNB()
	dt = DecisionTreeClassifier(random_state=0)
	rf = RandomForestClassifier(max_depth=20, random_state=0)
	lr = LogisticRegression(multi_class='multinomial',solver='newton-cg')
	
	# Fit the different machine learning models
	nb.fit(X_train, y_train)
	dt.fit(X_train, y_train)
	rf.fit(X_train, y_train)
	lr.fit(X_train, y_train)
	
	# The baseline model generates (-1 or 1) randomly
	preds_bl = np.ones(len(y_test))-2*randint(0,2,len(y_test))
	preds_nb = nb.predict(X_test)
	preds_dt = dt.predict(X_test)
	preds_rf = rf.predict(X_test)
	preds_lr = lr.predict(X_test)
	
	print('================================================================\n')
	print('Train data set size: ', len(y_train),'\n')
	print('Test data set size: ', len(y_test),'\n')
	print('================================================================\n')
	print("Baseline Model: \n",classification_report(y_test,preds_bl))
	print('================================================================\n')
	print("Naive Bayes: \n" ,classification_report(y_test,preds_nb))
	print('================================================================\n')
	print("Desicion Tree: \n",classification_report(y_test,preds_dt))
	print('================================================================\n')
	print("Random Forests: \n",classification_report(y_test,preds_rf))
	print('================================================================\n')
	print("Logistic Regression: \n",classification_report(y_test,preds_lr))
	print('================================================================\n')
	
	#--------------------------------------------------------------------------
	# Save the models
	# Here we save the model
	terms = vectorizer.get_feature_names()
	
	filename1 = 'source/lr_sentiment_model.sav'
	filename2 = 'source/nb_sentiment_model.sav'
	pickle.dump(lr, open(filename1, 'wb'))
	pickle.dump(nb, open(filename2, 'wb'))
	joblib.dump((X,terms,dict_sample), "source/articles-raw.pkl")
	
	print('Models have been trained and Saved in the Source File!')
	
