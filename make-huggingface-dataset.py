# pip install transformers datasets huggingface-hub accelerate evaluate tensorboard

from datasets import load_dataset
audio_dataset = load_dataset("audiofolder", data_dir = "samples/st-pauls-scott")
audio_dataset.push_to_hub("nacnudus/psalter")
