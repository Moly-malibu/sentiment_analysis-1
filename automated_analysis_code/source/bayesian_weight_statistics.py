'''
This code combines the sentiment models using Bayesian weights
'''





import warnings
import numpy as np

warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from source.process_text import sentiment_model
from source.process_text import clean_up_text
from source.process_text import preprocess
from source.process_text import string_cohesion

	
import warnings
warnings.filterwarnings('ignore')

from textblob import TextBlob
import spacy
import pickle
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
	



def compute_model_statistics(threshold):
	# Specify the name of the file, along with the name of the columns
	#file_name = './sentiment140/training.1600000.processed.noemoticon.csv'
	file_name = 'training_data/testdata.manual.2009.06.14.csv'
	
	cols = ['sentiment','id','date','query_string','user','text']
	
	# Read in the data, randomly shuffle and reset the index
	df=pd.read_csv(file_name,header=None, names=cols,encoding = "ISO-8859-1")
	df=df.sample(frac=1)
	df = df[df.sentiment != 2]
	df=df.reset_index(drop=True)
	
	# clean up the text
	df['text']= df['text'].apply(clean_up_text)
	
	
	# Load in the sentiment models 
	#--------------------------------------------------------------
	# Import the custom stop words
	custom_stop_words = []
	with open( "source/stopwords.txt", "r" ) as fin:
	    for line in fin.readlines():
	        custom_stop_words.append( line.strip() )
	#--------------------------------------------------------------
	
	
	#--------------------------------------------------------------
	# Import the vocabulary and generate the vectorizer transformer
	#--------------------------------------------------------------
	(A,terms,dict_sample) = joblib.load("source/articles-raw.pkl" )
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


	# Generate model data for different thresholds
	threshold = np.arange(0.0,1.05,0.05)
	model_accuracy = []
	model_sample_size = []
	model_recall=[]
	model_f1=[]
	
	model_true_pos_percent = []
	model_true_neg_percent = []
	model_false_pos_percent =[]
	model_false_neg_percent =[]
	
	model_true_pos = []
	model_true_neg = []
	model_false_pos =[]
	model_false_neg =[]
	
	
	for t in threshold:
	    
	    sentiment_pred = []
	    sentiment_prob = []
	    y = []
	    
	    for k in tqdm(range(0,len(df))):
	        # The Tweet to be analysed
	        sample_text = df['text'][k]
	
	        # The labelled sentiment of the tweet
	        true_sentiment = df['sentiment'][k]
	        
	
	        # We map the sentiment to our values
	        if(true_sentiment==0):
	            y_true = -1
	        elif(true_sentiment==4):
	            y_true = +1
	
	        pred,prob = sentiment_model(sample_text,t,vectorizer, loaded_lr,loaded_nb)
	        
	        # We look only at non-neutral sentiment predictions
	        if(pred!=0):
	            y.append(y_true)
	            sentiment_pred.append(pred)
	            sentiment_prob.append(prob)
	    
	    # Compute the confusion matrix:
	    cm = confusion_matrix(y_true=y, y_pred=sentiment_pred)
	    cm_normed = cm/cm.sum(axis=1)[:, np.newaxis]
	    
	    # Confusion matrices percentages
	    model_true_pos.append(cm[1,1])
	    model_true_neg.append(cm[0,0])
	    model_false_pos.append(cm[1,0])
	    model_false_neg.append(cm[0,1])
	    
	     # Confusion matrices percentages
	    model_true_pos_percent.append(cm_normed[1,1])
	    model_true_neg_percent.append(cm_normed[0,0])
	    model_false_pos_percent.append(cm_normed[1,0])
	    model_false_neg_percent.append(cm_normed[0,1])
	    
	    # Compute the accuracy and the recall of the models
	    acc = accuracy_score(y_pred=sentiment_pred,y_true=y) 
	    rec = recall_score(y_pred=sentiment_pred, y_true=y)
	    f1 = f1_score(y_pred=sentiment_pred, y_true=y)

	    model_accuracy.append(rec)
	    model_recall.append(acc)
	    model_f1.append(f1)
	    model_sample_size.append(len(y)*(1.0/len(df))*100.0) # Sample size is in percent
	

	plt.plot(threshold,model_true_pos_percent,"-o",label="True Positive")
	plt.plot(threshold,model_false_pos_percent,"-o",label="False Positive")
	plt.title('Positive Scores',size=20)
	plt.ylabel('Percentage',size=20)
	plt.xlabel('Threshold Value',size=20)
	plt.legend()
	plt.savefig('data/positive_rate_vs_threshold.pdf')
	
	plt.clf()
	plt.plot(threshold,model_true_neg_percent,"-o",label="True Negative")
	plt.plot(threshold,model_false_neg_percent,"-o",label="False Negative")
	plt.title('Negative Scores',size=20)
	plt.ylabel('Percentage',size=20)
	plt.xlabel('Threshold Value',size=20)
	plt.legend()
	plt.savefig('data/negative_rate_vs_threshold.pdf')
	
	plt.plot(threshold,model_sample_size,'o')
	plt.title('Sample Size vs Threshold',size=20)
	plt.ylabel('Sample Size (Percent)',size=20)
	plt.xlabel('Threshold Value',size=20)
	plt.savefig('data/sample_size_rate_vs_threshold.pdf')
	
	
	f_name_out = "data/sentiment_model_tweet_test_scores.txt"
	
	f = open(f_name_out,"w")
	
	f.write("    T    Acc    Rec     f1   size%    N   Tp_N     Fp_N    Tn_N    Fn_N    Tp%    Fp%    Tn%    Fn% "+"\n")
	
	for k in range(0,len(model_accuracy)):
	    t = threshold[k]
	    a = round(model_accuracy[k],2)
	    r = round(model_recall[k],2)
	    f1 = round(model_f1[k],2)
	    s = round(model_sample_size[k],2)
	    
	    tp = round(model_true_pos[k],2)
	    tn = round(model_true_neg[k],2)
	    fp = round(model_false_pos[k],2)
	    fn = round(model_false_neg[k],2)
	    
	    tp_percent = round(model_true_pos_percent[k],2)
	    tn_percent = round(model_true_neg_percent[k],2)
	    fp_percent = round(model_false_pos_percent[k],2)
	    fn_percent = round(model_false_neg_percent[k],2)
	    
	    print("Threshold: %f, accuracy: %f, recall: %f, sample size: %f" % (t,a,r,s))
	    line="%.4f %.4f %.4f %.4f %.4f %.0f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n"
	    f.write(line%(t,a,r,f1,s,tp+fp,tp,fp,tn,fn,tp_percent,fp_percent,tn_percent,fn_percent))
	    
	f.close()
	
