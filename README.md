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
