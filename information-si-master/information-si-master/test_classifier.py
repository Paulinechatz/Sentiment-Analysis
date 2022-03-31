#!/usr/bin/env python3
from palo import PaloDataset
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import argparse as ap
import pandas as pd

parser = ap.ArgumentParser(description='Test a sentiment classifier')

parser.add_argument('-v', '--vocab', type=str, help='Vocabulary')
parser.add_argument('-c', '--classifier', type=str, help='Classifier')
parser.add_argument('-ts', '--test-dataset', type=str, help="Palo test dataset (in csv format)")
parser.add_argument('--text', type=str, help='text field in the datasets', default='text')
parser.add_argument('--sentiment', type=str, help='sentiment field in the datasets', default='sentiment')

args = parser.parse_args()

df_test = pd.read_csv(args.test_dataset)

# Drop nan raws due to preprocessing
df_test = df_test[~df_test[args.text].isna()]


test_texts, test_labels = df_test[args.text].to_list(), df_test[args.sentiment].to_list()

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.vocab, max_len=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)
test_dataset = PaloDataset(test_encodings, test_labels)

model = AutoModelForSequenceClassification.from_pretrained(args.classifier)
trainer = Trainer(model=model)

results = trainer.predict(test_dataset)
preds = [i.index(max(i)) for i in results.predictions.tolist()]
print(classification_report(test_labels, preds, digits=4))
