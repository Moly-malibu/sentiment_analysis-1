import numpy as np
import time
import csv
import sqlite3
import json
import Programs.gps_geometry as gps
import Programs.twython_streamer as streamer
import sys

# Connect to the database
conn = sqlite3.connect("Databases/keyword_based_database.db")

# Print off the keywords that we use
with open('track.csv', newline='') as csvfile:
    trackreader = csv.reader(csvfile, delimiter=';', quotechar='|')
    for row in trackreader:
        trackstring = ', '.join(row)
        print(','.join(row))


# Load credentials from json file
with open("Credentials/twitter_credentials.json", "r") as file:  
    creds = json.load(file)


# Instantiate from our streaming class
stream = streamer.MyStreamer(conn,creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'],  
                    creds['ACCESS_TOKEN_KEY'], creds['ACCESS_TOKEN_SECRET'],
                   queryText=trackstring)


print("--------Keyword based listener starting--------")

# Start the stream, this fix was suggested from the following github page:
# https://github.com/ryanmcgrath/twython/issues/288 
while True:
	try:
		stream.statuses.filter(track=trackstring)  
	except:
		e = sys.exc_info()[0]
		print("error",e)
		time.sleep(20.0)
		continue
