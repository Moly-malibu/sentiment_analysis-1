import numpy as np
import csv
import sqlite3
import json
import Programs.gps_geometry as gps
import Programs.twython_streamer as streamer

# Connect to the database
conn = sqlite3.connect("Databases/location_based_database.db")

# Print off the keywords that we use
with open('track.csv', newline='') as csvfile:
    trackreader = csv.reader(csvfile, delimiter=';', quotechar='|')
    for row in trackreader:
        trackstring = ', '.join(row)
        print(','.join(row))


# Load credentials from json file
with open("Credentials/twitter_credentials.json", "r") as file:  
    creds = json.load(file)

# Coordinates of berlin
lon, lat, R =-0.076132, 51.508530, 50.0
geo_locations = gps.bounding_square_coordinates(lon,lat,R)

# Instantiate from our streaming class
stream = streamer.MyStreamer(conn,creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'],  
                    creds['ACCESS_TOKEN_KEY'], creds['ACCESS_TOKEN_SECRET'],
                   long=lon,lat=lat,Radius=R)
# Start the stream
stream.statuses.filter(locations=geo_locations)  

print("--------location based listener started--------")
