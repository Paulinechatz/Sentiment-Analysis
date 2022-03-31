#!/usr/bin/env python3
from palo import PaloDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.trainer_utils import IntervalStrategy
import pandas as pd
import argparse as ap

parser = ap.ArgumentParser(description='Train a sentiment classifier')

parser.add_argument('-v', '--vocab', type=str, help='Vocabulary ')
parser.add_argument('-lm', '--language-model', type=str, help='Language model (name or path)')
parser.add_argument('-c', '--classifier', type=str, help='Classifier location (where to store)')
parser.add_argument('-tr', '--train-dataset', type=str, help="Palo train dataset (in csv format)")
parser.add_argument('-val', '--val-dataset', type=str, help="Palo validation dataset (in csv format)")

parser.add_argument('--text', type=str, help='text field in the datasets', default='text')
parser.add_argument('--sentiment', type=str, help='sentiment field in the datasets', default='sentiment')

parser.add_argument('--train-dir', type=str, help='Training directory', default='./results')
parser.add_argument('--log-dir', type=str, help='Log directory', default='./logs')

parser.add_argument('--epochs', type=int, help='Number of epochs', default=3)
parser.add_argument('--batch-size', type=int, help='Batch size (training & evaluation', default=8)
parser.add_argument('--warmup', type=int, help='Warmup steps', default=500)
parser.add_argument('--wd', type=float, help='Weight decay', default=0.01)

args = parser.parse_args()

df_train = pd.read_csv(args.train_dataset)
df_val = pd.read_csv(args.val_dataset)

# Drop nan raws due to preprocessing
df_train = df_train[~df_train[args.text].isna()]
df_val = df_val[~df_val[args.text].isna()]

train_texts, train_labels = df_train[args.text].to_list(), df_train[args.sentiment].to_list()
val_texts, val_labels = df_val[args.text].to_list(), df_val[args.sentiment].to_list()

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.vocab, max_len=512)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

train_dataset = PaloDataset(train_encodings, train_labels)
val_dataset = PaloDataset(val_encodings, val_labels)

training_args = TrainingArguments(
    output_dir=args.train_dir,          # output directory
    num_train_epochs=args.epochs,              # total number of training epochs
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
    warmup_steps=args.warmup,                # number of warmup steps for learning rate scheduler
    weight_decay=args.wd,               # strength of weight decay
    logging_dir=args.log_dir,            # directory for storing logs
    evaluation_strategy=IntervalStrategy.EPOCH,
    logging_strategy=IntervalStrategy.EPOCH,
    load_best_model_at_end=True  # save_strategy=IntervalStrategy.EPOCH,
)

model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model,
                                                           num_labels=3)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()
trainer.save_model(args.classifier)
