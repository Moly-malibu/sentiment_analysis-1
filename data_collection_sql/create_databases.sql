PRAGMA encoding="UTF-8";
CREATE TABLE IF NOT EXISTS tweet_log (
  tweet_log_id INTEGER PRIMARY KEY AUTOINCREMENT,
  query text DEFAULT NULL,
  geo_lat real DEFAULT NULL,
  geo_long real DEFAULT NULL,
  radius real default NULL,
  timestamp_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

PRAGMA encoding="UTF-8";
CREATE TABLE IF NOT EXISTS user (
  user_id_text text primary key,
  screen_name text NOT NULL,
  name text DEFAULT NULL,
  location text DEFAULT NULL,
  url text DEFAULT NULL,
  description text DEFAULT NULL,
  created_at datetime NOT NULL,
  followers_count integer DEFAULT NULL,
  friends_count integer DEFAULT NULL,
  statuses_count integer DEFAULT NULL,
  time_zone text DEFAULT NULL
);

PRAGMA encoding="UTF-8";
CREATE TABLE IF NOT EXISTS tweet (
  tweet_log_id integer not NULL,
  tweet_id_text text primary key,
  tweet_hashtag text NOT null,
  tweet_text text NOT NULL,
  created_at datetime NOT NULL,
  geo_lat real DEFAULT NULL,
  geo_long real DEFAULT NULL,
  user_id_text text NOT NULL,
);
