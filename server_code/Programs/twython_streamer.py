from twython import TwythonStreamer  


# Create a class that inherits TwythonStreamer
# The class constructor has been overidden so it can have extra properties
class MyStreamer(TwythonStreamer): 
	
	# The class constructor
	def __init__(self,conn,app_key, app_secret, oauth_token, oauth_token_secret,queryText='',long='',lat='',Radius=''):
		TwythonStreamer.__init__(self,app_key, app_secret, oauth_token, oauth_token_secret)
		self.queryText = queryText
		self.long = long
		self.lat = lat
		self.Radius = Radius
		self.conn = conn
		self.cursor = conn.cursor()

	# Received data
	def on_success(self, data):

		# Only collect tweets in English
		try:
			if data['lang'] == 'en':
				tweet_data = process_tweet(data)
				tweet_id_text =  data['id_str']
				user_id = data['user']['id_str']
				time_stamp = data['created_at']
				self.save_to_sql(tweet_data, tweet_id_text, user_id, time_stamp)
		except:
			print("Unexpected error:", sys.exc_info()[0])
			pass

	# Problem with the API
	def on_error(self, status_code, data):
		print(status_code, data)
		self.disconnect()

	# Save each tweet to sql database
	def save_to_sql(self, tweet, tweet_id_text, user_id, time_stamp):
		
		# Redifine variable for simplicity of the function
		cursor = self.cursor
		conn = self.conn
		
		#tweet_log_id = tweet_id_text
		query_text = self.queryText
		query_long = self.long
		query_lat = self.lat
		query_radius = self.Radius
		
		# tweet_log_id, query text, geo_lat, geo_long, radius, timestamp
		query_data = [tweet_id_text,query_text,query_lat,query_long,query_radius, time_stamp]
			 
		# We ask the database if the given user already exist
		query ='''
		SELECT count(*) FROM user WHERE user_id_text=?
		'''
		cursor.execute(query, (user_id,))
		
		
		# Save the User Data, if the user is Unique
		if cursor.fetchone()[0] == 0:
			
			usr_query ='''
			INSERT INTO user(user_id_text, screen_name, location, url, description, created_at,
			followers_count, friends_count, statuses_count, time_zone) VALUES (?,?,?,?,?,?,?,?,?,?)
			'''
			cursor.execute(usr_query,tweet[1])
			
		tweet_query='''
		INSERT INTO tweet(tweet_id_text,tweet_hashtag,tweet_text,created_at,
		geo_lat, geo_long, user_id_text) VALUES (?,?,?,?,?,?,?)
		'''
		cursor.execute(tweet_query,tweet[0])
			
		log_query='''
		INSERT INTO tweet_log(tweet_id_text,query,geo_lat,geo_long,radius,timestamp_at) VALUES (?,?,?,?,?,?)
		'''
		cursor.execute(log_query, query_data)
		
		# Commit the data to the file
		conn.commit()
		
		
# Filter out unwanted data
def process_tweet(tweet):
	  
	tweet_id_str = tweet['id_str']
	tweet_text = tweet['text']
	tweet_created_at = tweet['created_at']
	tweet_user_idstr= tweet['user']['id_str']
	tweet_hash_tags = str(tweet['entities']['hashtags'])
	
	 
	if(tweet["geo"]!= None):
		tweet_geo_coords = tweet["geo"]["coordinates"]
		tweet_geo_lat =  tweet_geo_coords[0]
		tweet_geo_lon =  tweet_geo_coords[1]
	else:
		tweet_geo_coords = None
		tweet_geo_lon = None
		tweet_geo_lat = None
		

	# User information
	tweet_user_id_str = tweet['user']['id_str']
	tweet_user_screen_name =  tweet['user']['screen_name']
	tweet_user_location = tweet['user']['location']
	tweet_user_url =  tweet['user']['url']
	tweet_user_description = tweet['user']['description']
	tweet_user_created_at = tweet['user']['created_at']
	tweet_user_followers_count =tweet['user']['followers_count']
	tweet_user_friends_count =tweet['user']['friends_count']
	tweet_user_statuses_count =tweet['user']['statuses_count']
	tweet_user_time_zone = None
	
	if(tweet['place']!=None):
			tweet_user_time_zone = tweet['place']['country_code']
	#else:
	#	tweet_user_time_zone = None
	
	
	
	# Tweet Data: (8 items)
	# tweet_log_id, tweet_id_text, tweet_hashtag , tweet_text, created_at,  geo_lat,  geo_long , user_id_text
	d1 = [tweet_id_str , tweet_hash_tags,tweet_text,tweet_created_at,tweet_geo_lat,tweet_geo_lon, tweet_user_idstr]
	

	# User Data: (10 items) 
	# user_id_text, screen_name, location, url,  description text,
	# created_at, followers_count ,  friends_count ,statuses_count , time_zone 
	d2 = [tweet_user_id_str, tweet_user_screen_name,tweet_user_location,tweet_user_url,
		  tweet_user_description, tweet_user_created_at,tweet_user_followers_count ,tweet_user_friends_count,
		 tweet_user_statuses_count,tweet_user_time_zone]
	
	return [d1, d2]
