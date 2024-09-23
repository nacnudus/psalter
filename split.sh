#!/bin/bash

# Split audio files into segments at the given timestamps.
#
# Usage:
# ./split.sh recordings/st-pauls-scott/disc01/track01.flac
#
# Effect: creates the following files
# audio/st-pauls-scott/disc01/track01/segment001.flac
# audio/st-pauls-scott/disc01/track01/segment002.flac
# etc.

recording="$1"

# Infer the location of the timestamps
timestamps="timestamps/${recording#*/}" # Replace the top directory
timestamps="${timestamps%.*}.txt" # Replace the file extension

# Construct an output directory path
audio="audio/${recording#*/}" # Replace the top directory
audio="${audio%.*}"       # Remove the file extension, so that it could be a directory

# Create the path in case it doesn't aready exist
mkdir -p $audio

# Construct a comma-separated list of timestamps to split at, in seconds
segment_times=$(awk '{print $1}' "$timestamps" | paste -s -d',')

# Split the audio file into segments, and save those to the audio directory
ffmpeg -loglevel 24 -vn -sn -dn -i "$1" -codec copy -f segment -segment_list segments.csv -segment_times $segment_times "$audio/segment%03d.flac"
