# Twitter Data Streamer Collector

This folder contains all of the code that is used in order to stream and collect tweets,
either by keywords, or by a specific location.

In the __Credentials__ folder, make sure that you have a .json file with the following information:

```
{
    "CONSUMER_KEY": "your_consumer_key",
    "CONSUMER_SECRET": "your_consumer_secret",
    "ACCESS_TOKEN_KEY": "your_access_token_key",
    "ACCESS_TOKEN_SECRET": "your_access_token_secret"
}
```

## Getting Started

To create the data bases and run the python code run:

```
$ bash  create_tables_and_listen.sh
```

The sql databases will be stored in the folder __Databases__, as
* keyword_based_database.db
* location_based_database.db


