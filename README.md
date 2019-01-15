# Sentiment Analysis of Automotive Companies
 
This repository carries out sentiment analysis on website review data from different sources. Please look at the main [Jupyter notebook](Review_Analysis.ipynb) 


This code was written for a sentiment analysis model for [__APIthinking__](https://www.apithinking.de/en/).

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

### 4. Export changes to conda enviroment  
If you have made changes to the enviroment needed to run the code in the repository, export the enviroment using,  
```
$conda env export > <environment-name>.yml
```

# Repository contents

### 1. [Sentiment analysis model](sentiment_model/)  
This model was trained using a combination of Amazon customer review data along with Yelp review data.
The final chosen model was a Naive Bayes Classifier and Logistic regression.


### 2. [Twitter data collection code](server_code/)
This folder contains all of the code required to run the Twitter data listener and collect 
the twitter data as a SQL database.
