# Now we need to process the text. We define here a function that will remove the punctuation and stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
import operator
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter 
import spacy
from textblob import TextBlob


# Load the pre-built word embedding model
#nlp=spacy.load('/home/javier/anaconda3/envs/py35/lib/python3.5/site-packages/spacy/data/en_core_web_lg') # In case we need to download it
nlp = spacy.load('en_core_web_lg')

# In case we dont have the nltk stopwords documents
nltk.download('wordnet')
nltk.download("stopwords")
nltk.download('punkt')

ps = PorterStemmer()
wnl = WordNetLemmatizer()

# Load in the custom stop-words
custom_stop_words = []
with open( "stopwords.txt", "r" ) as fin:
    for line in fin.readlines():
        custom_stop_words.append( line.strip() )

@np.vectorize
def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    #words = filter(lambda x: x not in stopwords.words('english'), tokens)
    words = filter(lambda x: x not in custom_stop_words, tokens)
    s=str(" ".join(words))
    return s


def get_pos( word ):
    '''
    Part-Of-Speech Tagger
    '''
    w_synsets = wordnet.synsets(word)
    
    
    # n-noun, v-verb, a-adjective, r-
    pos_counts = Counter()
    pos_counts["n"] = len(  [ item for item in w_synsets if item.pos()=="n"]  )
    pos_counts["v"] = len(  [ item for item in w_synsets if item.pos()=="v"]  )
    pos_counts["a"] = len(  [ item for item in w_synsets if item.pos()=="a"]  )
    pos_counts["r"] = len(  [ item for item in w_synsets if item.pos()=="r"]  )
    
    most_common_pos_list = pos_counts.most_common(3)
    return most_common_pos_list[0][0]

def clean_up_text(text):
    '''
    This function will preprocess a text string, remove punctuation ect
    to make it easier to process.
    '''
    
    # Remove @ from the string
    s=re.sub(r'@[A-Za-z0-9]+','',text)    
    
    # Remove URLS
    s=re.sub('https?://[A-Za-z0-9./]+','',s)
    
    # Set to lower case
    s=s.lower()
    
    # Remove numbers
    s= re.sub(r'[0-9]+', '', s) 
    
    # Remove punctuation
    s = re.sub(r'[^\w\s]','',s)
    
    # Remove underscore
    s=s.replace("_", " ")
    
    # Remove RT from the tweet
    s=s.replace("rt", "")
    
    # Tokenize the words
    words = word_tokenize(s)
    
    # Lemmatize the words
    s_stem = ''
    for wi in words:
        #wi = ps.stem(wi)
        wi = wnl.lemmatize(wi,get_pos(wi))
        s_stem += ' ' + wi
    
    # Remove the stop words
    s_stem = preprocess(s_stem)
    
    return s_stem

def string_cohesion(vec):
	'''
	This function will compute the cohesion of a vector of words. This function
	is useful for choosing the number of topics for topic modelling
	
	
	The function has the following bounds:
	
	0<= string_cohesion(x) <=1
	
	string_cohesion(x) = 1
	when x consists of the same words
	
	string_cohesion(x) = 0
	when x is an empty string or there are no tokens for it
	'''
	s=0.0
	
	tokens = nlp(vec)
	Norm = len(tokens)
	
	# If we have a non-zero norm
	if(Norm!=0):
		
		for word1 in tokens:
			for word2 in tokens:
				if(word1.has_vector==True and word2.has_vector == True):
					s+= word1.similarity(word2)	
		s/=Norm**2
	
	return s

def sentiment_model(text,threshold,vectorizer, loaded_model1,loaded_model2):
    
    s = str(clean_up_text(text))
    
    X0 = vectorizer.transform([s])
    
    preds_nb = loaded_model1.predict(X0)
    preds_lr = loaded_model2.predict(X0)
    preds_blob =  TextBlob(text).sentiment.polarity
    
    if(preds_blob>=0.0):
        preds_blob = +1
    else:
        preds_blob = -1
        
    indx_nb= int((preds_nb[0]+1)/2)
    indx_lr= int((preds_lr[0]+1)/2)
    prob_nb = abs((loaded_model1.predict_proba(X0)[0][indx_nb]-.50)*2.0*int(preds_nb[0]))
    prob_lr = abs((loaded_model2.predict_proba(X0)[0][indx_lr]-0.50)*2.0*int(preds_lr[0]))
    prob_blob = abs(TextBlob(text).sentiment.polarity) # [-1,1]
    
    
    if(prob_lr<threshold):
        preds_lr=[0.0]
        
    if(prob_nb<threshold):
        preds_nb=[0.0]
        
    if(prob_blob<threshold):
        preds_blob =0.0
        
        
    # choose the most likely model
    predictions = [preds_nb[0],preds_lr[0],preds_blob]
    predict_prob = [prob_nb,prob_lr,prob_blob]
    
    pred = predictions[np.argmax(predict_prob)]
    prob = np.max(predict_prob)
    
    
    return pred,prob
