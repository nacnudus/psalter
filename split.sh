#!/bin/bash

# Split ripped CD tracks into segments at the given timestamps.
#
# Usage:
# ./split.sh recordings/st-pauls-scott/disc01/01.flac timestamps/st-pauls-scott/disc01/01.txt
#
# Effect: creates the following files
# audio/st-pauls-scott/disc01/track01/subtrac001.flac
# audio/st-pauls-scott/disc01/track01/subtrac002.flac
# etc.

recording="$1"
timestamps="$2"

# Construct an output directory
audio="${recording}"
audio="${audio#*/}"            # Remove the first directory
audio="${audio%.*}"            # Remove the file extension
track=$(basename "$audio")
audio=$(dirname "$audio")       # Remove the track number
audio="${audio}/track${track}" # Reinstate the track number
audio="audio/${audio}" # Prepend a new first directory
mkdir -p $audio

# Construct a comma-separated list of timestamps to split at, in seconds
segment_times=$(awk '{print $1}' "$timestamps" | paste -s -d',')

# Split the audio file into segments, and save those to the audio directory
ffmpeg -i "$1" -codec copy -f segment -segment_list segments.csv -segment_times $segment_times "$audio/subtrack%03d.flac"
