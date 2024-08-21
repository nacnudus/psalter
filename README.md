# Psalter

One day this might be a smartphone app similar to the website
https://www.rmjs.co.uk/psalter/psalter.php.

So far it's merely a JSON representation of the psalms.

## Wordlist

https://dhanswers.ach.org/topic/creating-a-wordlist-from-text/#post-1762

```sh
tr -sc '[A-Z][a-z]' '[\012*]' < psalmtext.txt | sort | uniq > wordlist.txt
```
