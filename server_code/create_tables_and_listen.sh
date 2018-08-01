#!/bin/bash

# Create the location based data base and run the script
echo "creating location based database"
sqlite3 Databases/location_based_database.db << EOF
.read create_databases.sql
EOF

echo "creating keyword based database"
sqlite3 Databases/keyword_based_database.db << EOF
.read create_databases.sql
EOF

echo "beginning python scripts"
#python location_listener.py &
#python keyword_listener.py &

