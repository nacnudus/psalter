# Create a JSON manifest of each audio file and its transcription.
#
# Also concatenate parts of a verse into a single file, as long as the total
# length is within the limits of the Whisper model, which is 30 seconds. Doing
# so is more efficient, because it will halve the amount of samples to use in
# training, and it mitigates the inaccuracy of splitting parts of verses that,
# when sung, did not pause at the colon.

library(dplyr)
library(purrr)
library(tidyr)
library(readr)
library(stringr)
library(sys)
library(fs)
library(jsonlite)

csv <- read_csv("./manifests/st-pauls-scott/manifest.csv")

clean <-
  csv %>%
  arrange(audio_file_path) %>%
  filter(!is.na(text)) %>%
  mutate(track = substr(audio_file_path, 1, 36)) %>%
  group_by(track) %>%
  mutate(verse_number = (seq_len(n()) + 1) %/% 2)

long <-
  clean %>%
  group_by(track, verse_number) %>%
  filter(sum(duration) > 30) %>%
  ungroup() %>%
  mutate(
         audio_file_paths = audio_file_path,
         ) %>%
  select(audio_file_paths, audio_file_path, text)

short <-
  clean %>%
  group_by(track, verse_number) %>%
  filter(sum(duration) <= 30) %>%
  ungroup() %>%
  group_by(track, verse_number) %>%
  summarise(
            audio_file_paths = paste0(audio_file_path, collapse = " "),
            text = paste0(text, collapse = " : "),
            .groups = "drop"
  ) %>%
  select(audio_file_paths, text)

all <-
  bind_rows(long, short) %>%
  mutate(audio_file_path = paste0("samples/st-pauls-scott/train/", sprintf("%04d", row_number()), ".flac"))
tail(all$audio_file_path)
head(all$audio_file_path)

# Concatenate (or don't) the audio files, and put them into the "samples"
# folder.
catflac <- function(inputs, output) {
  exec_wait("sox", c(str_split_1(inputs, " "), output))
}
fs::dir_create("samples")
fs::dir_delete("samples")
fs::dir_create("samples/st-pauls-scott/train")
walk2(all$audio_file_paths, all$audio_file_path, catflac)

# Create a metadata file
all %>%
  mutate(file_name = path_rel(audio_file_path, "samples/st-pauls-scott")) %>%
  select(file_name, transcription = text) %>%
  stream_out(file("./samples/st-pauls-scott/metadata.jsonl"))
