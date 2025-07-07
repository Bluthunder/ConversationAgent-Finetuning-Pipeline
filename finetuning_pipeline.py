# lightning_mistral_pipeline.py

import os
from sklearn.metrics import precision_recall_fscore_support
from lightning import LightningModule, LightningDataModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from datasets import load_dataset, concatenate_datasets, load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW

class MultiTaskDataModule(LightningDataModule):
    def __init__(self, nlu_train_path, nlu_val_path, nlg_train_path, nlg_val_path, tokenizer, batch_size=4):
        super().__init__()
        self.nlu_train_path = nlu_train_path
        self.nlu_val_path = nlu_val_path
        self.nlg_train_path = nlg_train_path
        self.nlg_val_path = nlg_val_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage=None):
        nlu_ds = load_dataset("json", data_files=self.nlu_train_path, split="train")
        nlg_ds = load_dataset("json", data_files=self.nlg_train_path, split="train")

        def tokenize_nlu(example):
            prompt = f"<s>[INST] {example['text']} [/INST]"
            labels = (
                f"Intent: {', '.join(example['intents'])}\n"
                f"Sentiment: {', '.join(example['sentiment'])}\n"
                f"NER: {', '.join(example['entities'])}"
            )
            return self.tokenizer(prompt, text_target=labels, truncation=True, padding='max_length', max_length=512)

        def tokenize_nlg(example):
            prompt = f"<s>[INST] {example['instruction']}\n{example['context']} [/INST]"
            labels = example['response']
            return self.tokenizer(prompt, text_target=labels, truncation=True, padding='max_length', max_length=512)

        nlu_ds = nlu_ds.map(tokenize_nlu)
        nlg_ds = nlg_ds.map(tokenize_nlg)
        self.train_dataset = concatenate_datasets([nlu_ds, nlg_ds])

        self.nlu_val_dataset = None
        self.nlg_val_dataset = None

        if self.nlu_val_path:
            nlu_val_ds = load_dataset("json", data_files=self.nlu_val_path, split="train")
            self.nlu_val_dataset = nlu_val_ds.map(tokenize_nlu)

        if self.nlg_val_path:
            nlg_val_ds = load_dataset("json", data_files=self.nlg_val_path, split="train")
            self.nlg_val_dataset = nlg_val_ds.map(tokenize_nlg)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        dataloaders = []
        if self.nlu_val_dataset:
            dataloaders.append(DataLoader(self.nlu_val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True))
        if self.nlg_val_dataset:
            dataloaders.append(DataLoader(self.nlg_val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True))
        return dataloaders if dataloaders else None


class MultiTaskModel(LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bleu = load_metric("bleu")
        self.rouge = load_metric("rouge")
        self.intent_preds, self.intent_labels = [], []
        self.sentiment_preds, self.sentiment_labels = [], []
        self.ner_preds, self.ner_labels = [], []

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model.generate(batch['input_ids'], attention_mask=batch['attention_mask'], max_length=128)
        preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        if dataloader_idx == 0:
            for pred, label in zip(preds, labels):
                if "Intent:" in pred and "Intent:" in label:
                    self.intent_preds.append(pred.split("Intent:")[1].split("\n")[0].strip())
                    self.intent_labels.append(label.split("Intent:")[1].split("\n")[0].strip())
                if "Sentiment:" in pred and "Sentiment:" in label:
                    self.sentiment_preds.append(pred.split("Sentiment:")[1].split("\n")[0].strip())
                    self.sentiment_labels.append(label.split("Sentiment:")[1].split("\n")[0].strip())
                if "NER:" in pred and "NER:" in label:
                    self.ner_preds.append(pred.split("NER:")[1].strip())
                    self.ner_labels.append(label.split("NER:")[1].strip())

        elif dataloader_idx == 1:
            self.bleu.add_batch(predictions=[p.split() for p in preds], references=[[l.split()] for l in labels])
            self.rouge.add_batch(predictions=preds, references=labels)

    def on_validation_epoch_end(self):
        if self.intent_preds and self.intent_labels:
            prec, rec, f1, _ = precision_recall_fscore_support(self.intent_labels, self.intent_preds, average="weighted", zero_division=0)
            self.log_dict({"nlu/intent_precision": prec, "nlu/intent_recall": rec, "nlu/intent_f1": f1})

        if self.sentiment_preds and self.sentiment_labels:
            prec, rec, f1, _ = precision_recall_fscore_support(self.sentiment_labels, self.sentiment_preds, average="weighted", zero_division=0)
            self.log_dict({"nlu/sentiment_precision": prec, "nlu/sentiment_recall": rec, "nlu/sentiment_f1": f1})

        if self.ner_preds and self.ner_labels:
            prec, rec, f1, _ = precision_recall_fscore_support(self.ner_labels, self.ner_preds, average="weighted", zero_division=0)
            self.log_dict({"nlu/ner_precision": prec, "nlu/ner_recall": rec, "nlu/ner_f1": f1})

        if hasattr(self, "bleu") and hasattr(self, "rouge"):
            rouge_scores = self.rouge.compute()
            bleu_score = self.bleu.compute()
            self.log("nlg/bleu", bleu_score["bleu"])
            self.log("nlg/rougeL", rouge_scores["rougeL"].mid.fmeasure)

        self.intent_preds, self.intent_labels = [], []
        self.sentiment_preds, self.sentiment_labels = [], []
        self.ner_preds, self.ner_labels = [], []

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        return optimizer


def main():
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    nlu_train_path = "s3://your-bucket/nlu_train.jsonl"
    nlu_val_path = "s3://your-bucket/nlu_val.jsonl"
    nlg_train_path = "s3://your-bucket/nlg_train.jsonl"
    nlg_val_path = "s3://your-bucket/nlg_val.jsonl"


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_module = MultiTaskDataModule(
        nlu_train_path=nlu_train_path,
        nlu_val_path=nlu_val_path,
        nlg_train_path=nlg_train_path,
        nlg_val_path=nlg_val_path,
        tokenizer=tokenizer
    )
    model = MultiTaskModel(model_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="mistral-nlu-nlg-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        max_epochs=5,
        logger=True,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=16,
        accelerator="gpu",
        devices=1,
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=0.25
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
