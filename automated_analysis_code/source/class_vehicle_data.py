import datetime
import numpy as np

class Vehicle_data:
	
	
	def __init__(self):
		"""
		The constructor of the class for the vehicle data
		"""
		
		
		self.car_data = {}
		self.car_data["Threshold"]=0.0
		self.car_data["Dates"] = []
		self.car_data["Daily_sample_size"]= []
		self.car_data["Daily_pos_percent"] = []
		self.car_data["Daily_neg_percent"] = []
		self.car_data["Daily_neu_percent"] = []
		self.car_data["Total_sample_size"]=None
		self.car_data["Total_pos_percent"]=None
		self.car_data["Total_neg_percent"]=None
		self.car_data["Total_neu_percent"]=None
		self.car_data["NA_indices"]=[]
	
	def read_in_data(self,file_name,year="2018"):
		"""
		This function will read in all of the data from a file
		"""
		f = open(file_name,mode="r")
		
		
		# Skip the first line
		f.readline()
		
		# Split the second line, add the data to the object
		line = f.readline().split()
		self.car_data["Total_sample_size"]=line[0]
		self.car_data["Total_pos_percent"]=line[1]
		self.car_data["Total_neg_percent"]=line[2]
		self.car_data["Total_neu_percent"]=line[3]
		
		# Read in all of the data from the file
		for line in f:
		    tokens = line.split()
		    month = Vehicle_data.month_function(tokens[0])
		    dates = datetime.date(int(year),int(month),int(tokens[1]))
		    
		    self.car_data["Dates"].append(dates)
		    self.car_data["Daily_sample_size"].append(float(tokens[2]))
		    self.car_data["Daily_pos_percent"].append(float(tokens[3]))
		    self.car_data["Daily_neg_percent"].append(float(tokens[4]))
		    self.car_data["Daily_neu_percent"].append(float(tokens[5]))
		    
		# Convert to numpy arrays
		self.car_data["Daily_sample_size"]=np.asarray(self.car_data["Daily_sample_size"])
		self.car_data["Daily_pos_percent"]=np.asarray(self.car_data["Daily_pos_percent"])
		self.car_data["Daily_neg_percent"]=np.asarray(self.car_data["Daily_neg_percent"])
		self.car_data["Daily_neu_percent"]=np.asarray(self.car_data["Daily_neu_percent"])
	
	@staticmethod
	def month_function(date):
		
		token = date.lower()
		
		month_number = -1
		
		if(token=="jan" or token == "january"):
			month_number =1
		elif(token=="feb" or token == "february"):
			month_number =2
		elif(token=="mar" or token == "march"):
			month_number = 3
		elif(token=="apr" or token == "april"):
			month_number =4 
		elif(token=="may"):
			month_number = 5
		elif(token=="jun" or token=="june"):
			month_number = 6
		elif(token=="jul" or token=="july"):
			month_number = 7 		
		elif(token=="aug" or token=="august"):
			month_number=8
		elif(token=="sept" or token=="september" or token=="sep"):
			month_number=9
		elif(token=="oct" or token=="october"):
			month_number=10
		elif(token=="nov" or token=="november"):
			month_number=11
		elif(token=="dec" or token=="december"):
			month_number=12
			
		
		
		
		return month_number
	
	def fill_NA(self,data_NA=""):
		"""
		This function will fill in the missing data
		"""
		
		N = len(self.car_data["Dates"])
		mean_pos = self.car_data["Daily_pos_percent"][self.car_data["Daily_pos_percent"] != -1].mean()
		mean_neg = self.car_data["Daily_neg_percent"][self.car_data["Daily_neg_percent"] != -1].mean()
		mean_neu = self.car_data["Daily_neu_percent"][self.car_data["Daily_neu_percent"] != -1].mean()
		
		for i in range(N):
			pos = self.car_data["Daily_pos_percent"][i]
			neg = self.car_data["Daily_neg_percent"][i]
			neu = self.car_data["Daily_neu_percent"][i]
			
			if(pos==-1.0):
				self.car_data["Daily_pos_percent"][i] = mean_pos
				self.car_data["NA_indices"].append(i)
			
			if(neg==-1.0):
				self.car_data["Daily_neg_percent"][i] = mean_neg
			
			if(neu==-1):
				self.car_data["Daily_neu_percent"][i] = mean_neu
	
	
	@staticmethod			
	def fill_NA_array(array,data_NA=""):
		"""
		This function will will in NA values in an array with the mean of the array
		"""
		N = len(array)
		mean_val = array[array != -1.0 ].mean()
		
		for i in range(N):
			ai = array[i]
			
			if(ai==-1.0):
				array[i]=mean_val
		
		return array
	
	
	def return_sentiment_data(self):

		return self.car_data["Daily_pos_percent"],self.car_data["Daily_neg_percent"],self.car_data["Daily_neu_percent"]
		
	def daily_sentiment_return(self):
		"""
		
		"""
		N = len(self.car_data["Dates"])
		
		daily_pos_return = (self.car_data["Daily_pos_percent"][1:]/self.car_data["Daily_pos_percent"][0:-1])-1.0
		daily_neg_return = (self.car_data["Daily_neg_percent"][1:]/self.car_data["Daily_neg_percent"][0:-1])-1.0
		daily_neu_return = (self.car_data["Daily_neu_percent"][1:]/self.car_data["Daily_neu_percent"][0:-1])-1.0
		
		
		return daily_pos_return, daily_neg_return, daily_neu_return
	
	def scale_data_Ztransform(self):
		
		N = len(self.car_data["Dates"])
		
		mean_pos = self.car_data["Daily_pos_percent"].mean()
		mean_neg = self.car_data["Daily_neg_percent"].mean()
		mean_neu = self.car_data["Daily_neu_percent"].mean()
		
		std_pos = np.std(self.car_data["Daily_pos_percent"])
		std_neg = np.std(self.car_data["Daily_neg_percent"])
		std_neu = np.std(self.car_data["Daily_neu_percent"])
		
		y_pos = np.zeros(N)
		y_neg = np.zeros(N)
		y_neu = np.zeros(N)
		
		if(std_pos==0.0):
			std_pos = 1e10
		
		if(std_neg==0.0):
			std_neg=1e10
		
		if(std_neu==0.0):
			std_neu=1e10
		
		for i in range(N):
			y_pos[i] = (self.car_data["Daily_pos_percent"][i]-mean_pos)/std_pos
			y_neg[i] = (self.car_data["Daily_neg_percent"][i]-mean_neg)/std_neg
			y_neu[i] = (self.car_data["Daily_neu_percent"][i]-mean_neu)/std_neu
		
		return y_pos,y_neg,y_neu
	
	
	def scale_data_MinMaxTransform(self,Xmin=0.0,Xmax=1.0):
		
		N = len(self.car_data["Dates"])
		
		pos_Max = np.max(self.car_data["Daily_pos_percent"])
		neg_Max = np.max(self.car_data["Daily_neg_percent"])
		neu_Max = np.max(self.car_data["Daily_neu_percent"])
		
		pos_Min = np.min(self.car_data["Daily_pos_percent"])
		neg_Min = np.min(self.car_data["Daily_neg_percent"])
		neu_Min = np.min(self.car_data["Daily_neu_percent"])
		
		
		y_pos = np.zeros(N)
		y_neg = np.zeros(N)
		y_neu = np.zeros(N)
		
		for i in range(N):
			y_pos[i]= ((self.car_data["Daily_pos_percent"][i]-pos_Min)/(pos_Max-pos_Min))*(Xmax-Xmin)+Xmin
			y_neg[i]= ((self.car_data["Daily_neg_percent"][i]-neg_Min)/(neg_Max-neg_Min))*(Xmax-Xmin)+Xmin
			y_neu[i]= ((self.car_data["Daily_neu_percent"][i]-neu_Min)/(neu_Max-neu_Min))*(Xmax-Xmin)+Xmin
			
		
		
		return y_pos,y_neg,y_neu
		
	@staticmethod
	def return_model_training_results_prob(file_name):
		
		prob_true_pos_T=[]
		prob_false_pos_T=[]
		prob_true_neg_T=[]
		prob_false_neg_T=[] 
		
		T= []
		Tp_N=[]
		Tn_N=[]
		Fp_N=[]
		Fn_N=[]

		f=open(file_name,mode="r")
		f.readline()
		
		
		for line in f:
			tokens = line.split()
			T.append(float(tokens[0]))
			Tp_N.append(float(tokens[6]))
			Fp_N.append(float(tokens[7]))
			Tn_N.append(float(tokens[8]))
			Fn_N.append(float(tokens[9]))
			
		prob_tp = np.zeros(len(T))
		prob_fp = np.zeros(len(T))
		prob_pos = np.zeros(len(T))
		prob_tn = np.zeros(len(T))
		prob_fn = np.zeros(len(T))
		prob_neg =np.zeros(len(T))
			
		for k in range(len(T)):
			# The probability of a positive or negative result
			prob_pos[k] = (Tp_N[k]+Fn_N[k])/(Tp_N[k]+Fp_N[k]+Tn_N[k]+Fn_N[k])
			prob_neg[k] = (Tn_N[k]+Fp_N[k])/(Tp_N[k]+Fp_N[k]+Tn_N[k]+Fn_N[k])
			
			prob_tp[k] = Tp_N[k]/(Tp_N[k]+Fp_N[k])
			prob_fp[k] = Fp_N[k]/(Tp_N[k]+Fp_N[k])
			
			prob_tn[k] = Tn_N[k]/(Tn_N[k]+Fn_N[k])
			prob_fn[k] = Fn_N[k]/(Tn_N[k]+Fn_N[k])
			
					
		
		return T, prob_tp,prob_fp, prob_tn,prob_fn, prob_pos, prob_neg
	
