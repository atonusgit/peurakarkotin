#!/bin/bash

# Navigate to the "havainnot-timelapse" directory
cd ~/peurakarkotin/havainnot-timelapse

# Sort the folders by name and store them in an array
sorted_folders=($(ls | sort))

# Get the total number of folders
total_folders=${#sorted_folders[@]}

# Calculate the number of folders to delete
folders_to_delete=$((total_folders - 7))

# Delete the oldest folders, keeping the 7 newest ones
for (( i=0; i<$folders_to_delete; i++ )); do
  folder=${sorted_folders[$i]}
  echo "Deleting folder $folder"
  rm -rf "$folder"
done

cd ~/peurakarkotin

