import os
import json
from lightning import LightningModule, LightningDataModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger
from datasets import load_dataset, concatenate_datasets, load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, get_cosine_schedule_with_warmup

class MultiTaskDataModule(LightningDataModule):
    def __init__(self, nlu_train_path, nlg_train_path, nlg_val_path, tokenizer, batch_size=4):
        super().__init__()
        self.nlu_train_path = nlu_train_path
        self.nlg_train_path = nlg_train_path
        self.nlg_val_path = nlg_val_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage=None):
        nlu_ds = load_dataset("json", data_files=self.nlu_train_path, split="train")
        nlg_ds = load_dataset("json", data_files=self.nlg_train_path, split="train")
        self.val_dataset = load_dataset("json", data_files=self.nlg_val_path, split="train")

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
        self.val_dataset = self.val_dataset.map(tokenize_nlg)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)


class MultiTaskModel(LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bleu = load_metric("bleu")
        self.rouge = load_metric("rouge")

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("train_loss", outputs.loss, prog_bar=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("val_loss", outputs.loss, prog_bar=True)

        generated_ids = self.model.generate(batch["input_ids"], max_new_tokens=64)
        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        self.bleu.add_batch(predictions=[pred.split() for pred in preds], references=[[label.split()] for label in labels])
        self.rouge.add_batch(predictions=preds, references=labels)

        if batch_idx == 0:
            self.logger.experiment.log_text("sample_preds", preds[0])

    def on_validation_epoch_end(self):
        bleu_score = self.bleu.compute()["bleu"]
        rouge_score = self.rouge.compute()["rougeL"].mid.fmeasure

        self.log("bleu", bleu_score, prog_bar=True)
        self.log("rougeL", rouge_score, prog_bar=True)

        metrics = {"bleu": bleu_score, "rougeL": rouge_score}
        with open("nlg_metrics.json", "w") as f:
            json.dump(metrics, f)

        self.logger.experiment.log_artifact(
            local_path="nlg_metrics.json",
            artifact_path="metrics"
        )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-5, weight_decay=0.01)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=1000
        )
        return [optimizer], [scheduler]


def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    nlu_train_path = "s3://your-bucket/nlu_train.jsonl"
    nlg_train_path = "s3://your-bucket/nlg_train.jsonl"
    nlg_val_path = "s3://your-bucket/nlg_val.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_module = MultiTaskDataModule(
        nlu_train_path=nlu_train_path,
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

    logger = MLFlowLogger(
        experiment_name="Mistral-7B-NLU-NLG",
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    )

    trainer = Trainer(
        max_epochs=5,
        logger=logger,
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