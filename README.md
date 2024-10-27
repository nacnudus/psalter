# Psalter

One day this might be a smartphone app similar to the website
https://www.rmjs.co.uk/psalter/psalter.php.

So far it's merely a JSON representation of the psalms.

## Wordlist

https://dhanswers.ach.org/topic/creating-a-wordlist-from-text/#post-1762

```sh
tr -sc '[A-Z][a-z]' '[\012*]' < psalmtext.txt | sort | uniq > wordlist.txt
```

## Transcription and alignment

Review https://github.com/nacnudus/audio-verse-splitter

## Consolidate timestamps files

```sh
find audio/priory-1 -type f -name "timestamps.txt" | xargs -I {} bash -c 'cp "$1" temp/$(basename $(dirname "$1")).txt' -- {}
```

## Copy a directory structure without files

```sh
rsync -a --include '*/' --exclude '*' recordings/priory-1 "timestamps"
```

## Split audio files at timestamps

```sh
find recordings/st-pauls-scott -type f -name "*.flac" | xargs -I {} bash -c './split.sh $1' -- {}
```

## Calculate the actual durations of audio files

```sh
fd -I --exec sox {} -n stat 2>&1 | grep "Length" | awk '{print $3}' > durations.txt
```

## Create audio samples and metadata

Use script `manifest2metadata.R` to create a JSON manifest of each audio file
and its transcription.

Also concatenate parts of a verse into a single file, as long as the total
length is within the limits of the Whisper model, which is 30 seconds. Doing so
is more efficient, because it will halve the amount of samples to use in
training, and it mitigates the inaccuracy of splitting parts of verses that,
when sung, did not pause at the colon.

## Create a Huggingface dataset

https://me.sakana.moe/2024/09/03/a-complete-guide-to-fine-tuning-and-deploying-whisper-models/

If huggingface login and dataset upload hang, you might have to disable IPv6.

```sh
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1
```
