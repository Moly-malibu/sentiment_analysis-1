import warnings
import numpy as np

warnings.filterwarnings('ignore')

import collections
import os.path
import os
import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from source.read_input import  *
from source.process_text import clean_up_text
from source.process_text import preprocess
from source.process_text import string_cohesion
from source.process_text import sentiment_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import pickle
from textblob import TextBlob


def generate_sentiment_time_series(time_stamps,car_companies,thresholds,database):
	
	#--------------------------------------------------------------
	# Import the custom stop words
	custom_stop_words = []
	
	with open( "source/stopwords.txt", "r" ) as fin:
		for line in fin.readlines():
			custom_stop_words.append( line.strip() )
	#--------------------------------------------------------------
	
	#--------------------------------------------------------------
	# Import the vocabulary and generate the vectorizer tranformer
	#--------------------------------------------------------------
	(A,terms,dict_sample) = joblib.load( "source/articles-raw.pkl" )
	print( "Loaded %d X %d document-term matrix" % (A.shape[0], A.shape[1]) )
	print('number of terms: ',len(terms))#print(dic_sample[0:10])
	print('Dictionary: ',len(dict_sample))
	
	vectorizer = TfidfVectorizer(stop_words = custom_stop_words,min_df = 20)
	A = vectorizer.fit_transform(dict_sample)
	print('A: ', A.shape)
	#--------------------------------------------------------------
	
	
	#--------------------------------------------------------------
	# We load the trained models
	#--------------------------------------------------------------
	# The names of the files containing the weights of the model
	filename1 = 'source/lr_sentiment_model.sav'
	filename2 = 'source/nb_sentiment_model.sav'
	
	# Now we load in the trained models
	loaded_lr = pickle.load(open(filename1, 'rb'))
	loaded_nb = pickle.load(open(filename2, 'rb'))
	#--------------------------------------------------------------
	
	conn = sqlite3.connect(database)
	
	x_labels = [time_stamps[k].replace("%", "").replace("'","") for k in range(1,len(time_stamps))]
	print('Time Stamps: ', x_labels)
	print('Number of Dates: ', len(x_labels),'\n')
	
	# Now we make the new folders in our current directoy
	for threshold in thresholds:
	    path = "data/output_t"+str(int(100*threshold))+"/"
	    
	    try:
	        os.mkdir(path)
	    except:
	        print("path exists: ",path)
	       
	       
	for threshold in thresholds:
		for car_company in car_companies:
			
			file_name = 'sql_queries/'+car_company+'.txt'
			
			q = read_in_sql_query(file_name)
			
			output_q_name = 'sql_queries/'+car_company+'_temp.sql'
			
			pos_sentiment = []
			neg_sentiment = []
			neutral_sentiment = []

			daily_total = []
			daily_neutral=[]
			daily_pos=[]
			daily_neg=[]
			
			for date in tqdm(time_stamps):
				
				q_time = q + "AND created_at LIKE '"+ date +"' \n "
				
				
				df = pd.read_sql_query(q_time,conn)
				N_raw_tweets = len(df)
				
				percents = [-1.0,-1.0,-1.0]
				total_number = -1.0
				
				if(N_raw_tweets !=0 ):
					# Apply the 'Clean-up-text function to all tweets in the data frame'
					 df['tweet_text'] = df['tweet_text'].apply(clean_up_text)
					 
					 # # Remove duplicate tweets and reset the index
					 df.drop_duplicates(subset='tweet_text', keep='first', inplace=True)
					 df=df.reset_index(drop=True)
					 N_processed_tweets = len(df)
					 
					 print('\n')
					 print('=============================================================')
					 print('The number of tweets after removing duplicates: ', N_processed_tweets ,'\n')
					 print('Fraction removed: ', round(1.0-(N_processed_tweets/float(N_raw_tweets)),3) )
					 print('=============================================================')
					 
					 sentiment_pred = []
					 sentiment_prob = []
					 
					 for k in range(0,len(df)):
						 sample_text = df['tweet_text'][k]
						 pred,prob = sentiment_model(sample_text,threshold,vectorizer, loaded_lr,loaded_nb)
						 sentiment_pred.append(pred)
						 sentiment_prob.append(prob)
						 
					 matplotlib.rcParams['figure.figsize'] = 10, 9
					 matplotlib.rcParams['font.size'] = 15
					 
					 # Count the various sentiments
					 m_count= collections.Counter(sentiment_pred)
					 size_model=[m_count[0],m_count[1],m_count[-1]]
					 
					 # # Compute the total numbers
					 total_number = np.sum(size_model)
					 
					 # # Compute the Percentages
					 percents = (np.asarray([m_count[0],m_count[1],m_count[-1]])/total_number)*100.0
					 
					 print("Total Sample: ",total_number)
					 print("Neg/Pos ratio: ", m_count[-1]/(m_count[1]+1e-10))
					 
					 # # Create a circle for the center of the plot
					 my_circle=plt.Circle( (0,0), 0.6, color='white')
					 
					 # # create data
					 names='Neutral: '+str(int(percents[0]))+"%",'Positive: '+str(int(percents[1]))+"%", 'Negative: '+str(int(percents[2]))+"%"
					 
					 # Give color names
					 plt.clf()
					 plt.title(car_company+" : "+date.replace("%", "").replace("'",""),size=30)
					 plt.pie(size_model, labels=names, colors=['lightgrey','green','red'])
					 p=plt.gcf()
					 p.gca().add_artist(my_circle)
					 plt.savefig("data/output_t"+str(int(100*threshold))+"/"+car_company+"_"+date.replace("%", "").replace("'","").replace(" ","_")+".pdf",bboxes="tight")
				
			
				 # # Save the Daily values of positive and negative sentiments
				daily_total.append(total_number)
				daily_neutral.append(percents[0])
				daily_pos.append(percents[1])
				daily_neg.append(percents[2])
				 
				#==========================================================================================================
				# Write put all of the Data to a file
			filename = "data/output_t"+str(int(100*threshold))+"/"+car_company+"_daily_data.txt"
			file = open(filename, "w")
			s = "Time_Stamp"+"      "+"Daily_Total "+"       "+"Daily_pos"+"    "+"Daily_neg"+"    "+"Daily_neutral"+"\n"
			file.write(s)
			
			s =  ' All '+str(daily_total[0])+" "+str(daily_pos[0])+" "+str(daily_neg[0])+" "+str(daily_neutral[0])+"\n"
			file.write(s)
			
			for k in range(1,len(daily_pos)):
				s = time_stamps[k].replace("%", "").replace("'","")+str(daily_total[k])+" "+str(daily_pos[k])+" "+str(daily_neg[k])+" "+str(daily_neutral[k])+"\n"
				file.write(s)
			
			file.close()
			#==========================================================================================================
			#=========================================================================================================
			# Fill in Missing Values:
			#Remove the first entry which is full set
			daily_pos =  daily_pos[1:]
			daily_neg = daily_neg[1:]
			daily_neutral = daily_neutral[1:]
			 
			print(len(daily_pos), len(daily_neg),len(daily_neutral))
			 
			mean_pos = daily_pos[daily_pos!=-1.0].mean()
			mean_neg = daily_neg[daily_neg!=-1.0].mean()
			mean_neutral = daily_neutral[daily_neutral!=-1.0].mean()
			
			# Initialize the index
			x_indx =[]
			y_pos =[]
			y_neg = []
			y_neu =[]
			 
			# The Full index range 0-> set length
			x_full_indx =range(0,len(daily_neg))
			 
			for k in range(len(daily_neg)):
				dpos = daily_pos[k]
				dneg = daily_neg[k]
				dneu = daily_neutral[k]
				
				# replace the missing values with the mean values of the array
				if(dpos==-1.0):
					daily_pos[k] = mean_pos
					daily_neg[k] = mean_neg
					daily_neutral[k] = mean_neutral
				else:
					x_indx.append(k)
					y_pos.append(dpos)
					y_neg.append(dneg)
					y_neu.append(dneu)
		# #==========================================================================================================
		# ## Plot the Data and save the file
			x_full_indx =range(0,len(daily_neg))
			plt.clf()
			plt.title(car_company+": Daily Sentiment %")
			plt.ylabel("Percent %",size=20)
			plt.plot(daily_pos,"-o",c="g", mfc='none')
			plt.scatter(x_indx,y_pos,c='g',marker='o',label="Positive",)
			plt.plot(daily_neg,"-o",c="r", mfc='none')
			plt.scatter(x_indx,y_neg,c='r',marker='o',label="Negative")
			plt.xticks(x_full_indx, x_labels, rotation='vertical')
			plt.legend()
			plt.savefig("data/output_t"+str(int(100*threshold))+"/"+car_company+"_Daily_Sentiment"+".pdf",bbox_inches="tight")


