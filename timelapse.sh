#!/bin/bash

echo "Starting the script..."
orig=$(pwd)

# Define the directory containing your images
image_dir="/home/pi/peurakarkotin/havainnot-timelapse/$1"
echo "Image directory: $image_dir"

# Change to the image directory
cd "$image_dir"

echo "Creating sorted list of image files..."
# Create a sorted list of files based on modification time
ls -1tr | grep .jpg > sorted_list.txt

echo "Creating the final video..."
# Create the final video using ffmpeg
ffmpeg -f concat -safe 0 -i <(while IFS= read -r file; do echo "file '$PWD/$file'"; done < sorted_list.txt) -s 640x480 -vf "fps=25" -pix_fmt yuv420p $1.mp4

echo "Moving the output video to destination directory..."
# Move the output file to the desired directory
mv $image_dir/$1.mp4 /home/pi/peurakarkotin/todropbox/

echo "Cleaning up..."
# Remove the sorted_list.txt file
rm sorted_list.txt

echo "Script completed successfully!"
cd $orig
