import numpy as np


# Read the input file


def read_in_sql_query(file_name):
	'''
	This function will read in a sql query from a text file
	'''
	
	input_file = open(file_name)

	#sql_query="'''"+'\n'
	k=0
	for line in input_file:
		k+=1
	input_file.close()
	
	
	# Re-open the file
	input_file = open(file_name)
	
	sql_query =""
	for i in range(k-1):
		line = input_file.readline()
		sql_query+=line
	
	sql_query+=input_file.readline()

	#sql_query+="'''"
	input_file.close()
	
	return sql_query
	
def save_sql_query(query,output_file):
	'''
	Saves a set of string commands to a file
	'''
	f = open(output_file,'w')
	
	f.write(query)
	
	f.close()
	
	return None
