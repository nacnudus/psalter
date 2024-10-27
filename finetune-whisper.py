import sys
!{sys.executable} -m pip install transformers datasets huggingface-hub accelerate evaluate tensorboard librosa jiwer

from huggingface_hub import login
login()

# 商业转载请联系作者获得授权，非商业转载请注明出处。
# For commercial use, please contact the author for authorization. For non-commercial use, please indicate the source.
# 协议(License)：署名-非商业性使用-相同方式共享 4.0 国际 (CC BY-NC-SA 4.0)
# 作者(Author)：metricv
# 链接(URL)：https://me.sakana.moe/2024/09/03/a-complete-guide-to-fine-tuning-and-deploying-whisper-models/
# 来源(Source)：MetricVoid&#039;s Blog

# NOTE: The base model you want to finetune. Fine-tuning large-v3 model requires around 32GB of VRAM.
base_model = "openai/whisper-tiny"

# NOTE: Don't change this. Unless you were trying to fine-tuning for translate, and your dataset is tagged with English translated input.
language = "English"
task = "transcribe"

from datasets import load_dataset, DatasetDict

# ========== Load Dataset ==========
tl_dataset = DatasetDict()
tl_dataset["train"] = load_dataset("nacnudus/psalter", split="train")
# NOTE: If you have a test split, uncomment the following.
# tl_dataset["test"] = load_dataset("metricv/tl-whisper", "hi", split="test")

# ========== Load Whisper Preprocessor ==========

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor

feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)
tokenizer = WhisperTokenizer.from_pretrained(base_model, language=language, task=task)
processor = WhisperProcessor.from_pretrained(base_model, task=task)

# ========== Process Dataset ==========

from datasets import Audio

tl_dataset = tl_dataset.cast_column("audio", Audio(sampling_rate=16000))


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    # encode target text to label ids
    # NOTE: Here, the key "transcription" refers to the column that contains transcription we used in metadata.jsonl. Change this if you changed column names.
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch


tl_dataset = tl_dataset.map(
    prepare_dataset, remove_columns=tl_dataset.column_names["train"], num_proc=8
)

# ========== Load Whisper Model ==========

from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(base_model)
model.generation_config.task = task
model.generation_config.forced_decoder_ids = None

# ========== Fine-tune model ==========

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

import evaluate

metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-tiny-psalter",  # change to a repo name of your choice
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    num_train_epochs=2.0,
    # warmup_steps=500,
    # max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    do_eval=False,
    # eval_strategy="steps",    # NOTE: If you have a test split, you can uncomment this.
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    # save_steps=1000,
    # eval_steps=1000,
    logging_steps=5,
    report_to=["tensorboard"],
    load_best_model_at_end=False,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=tl_dataset["train"],
    # eval_dataset=tl_dataset["test"], # NOTE: If you have a test split, you can uncomment this.
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

trainer.train()

# ========== Save model ==========

trainer.save_model(output_dir="./whisper-tiny-psalter")
torch.save(model.state_dict(), f"{training_args.output_dir}/pytorch_model.bin")

# ========== Push model to HF hub ==========
# If you do not want to push the model to hub, or your training machine do not have Internet connection, comment this out.
from huggingface_hub import login
login()
trainer.push_to_hub("nacnudus/psalter")  # Change this.
