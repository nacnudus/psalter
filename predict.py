import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_id = "nacnudus/whisper-tiny-psalter"
model_id = "nacnudus/whisper-medium-psalter"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps="word",
)

# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]

# result = pipe(sample)

result = pipe("./recordings/st-pauls-scott/disc01/track01.flac")
print(result["text"])

pipe("./recordings/st-pauls-scott/disc01/track03.flac")["text"]

pipe("./samples/st-pauls-scott/train/0002.flac")["text"]
pipe("./samples/st-pauls-scott/train/0003.flac")["text"]

pipe("./samples/st-pauls-scott/train/0001.flac")["chunks"]
pipe("./samples/st-pauls-scott/train/0001.flac")["text"]

pipe("./audio/priory-1/disc02/track06/segment001.flac")["text"]
pipe("./audio/priory-1/disc02/track06/segment002.flac")["text"]
pipe("./audio/priory-1/disc02/track06/segment003.flac")["text"]
pipe("./audio/priory-1/disc02/track06/segment004.flac")["text"]

pipe("./audio/westminster-abbey-neary/disc02/track11/segment001.flac")["text"]
pipe("./audio/westminster-abbey-neary/disc02/track11/segment002.flac")["text"]
pipe("./audio/westminster-abbey-neary/disc02/track11/segment003.flac")["text"]
pipe("./audio/westminster-abbey-neary/disc02/track11/segment004.flac")["text"]

pipe("./audio/kings-college-cambridge-willcocks/disc02/track06/segment000.flac")["text"]
pipe("./audio/kings-college-cambridge-willcocks/disc02/track06/segment001.flac")["text"]
pipe("./audio/kings-college-cambridge-willcocks/disc02/track06/segment002.flac")["text"]
pipe("./audio/kings-college-cambridge-willcocks/disc02/track06/segment003.flac")["text"]

pipe("./recordings/westminster-abbey-neary/disc01/track01.flac")["text"]
pipe("./recordings/kings-college-cambridge-willcocks/disc01/track01.flac")["text"]

pipe("./recordings/priory-1/disc01/track01.flac")["text"]
