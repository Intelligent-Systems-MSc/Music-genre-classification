# Download GTZAN dataset 
wget http://opihi.cs.uvic.ca/sound/genres.tar.gz

# Extract GTZAN dataset
tar -xvzf genres.tar.gz

# Remove tar file
rm genres.tar.gz

# Create data folder if it doesn't exist
if [ ! -d "data" ]; then
    mkdir Data
fi

# Move genre to data folder 
mv genres Data/

# Delete the files in genres 
rm -f data/*