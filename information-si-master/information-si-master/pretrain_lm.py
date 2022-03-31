#!/usr/bin/env python3
from tempfile import NamedTemporaryFile
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, LineByLineTextDataset, RobertaConfig, \
    AutoModelForMaskedLM, Trainer, TrainingArguments
from transformers.trainer_utils import IntervalStrategy
import argparse as ap
import pandas as pd

parser = ap.ArgumentParser(description='Train a pre-trained language model')

parser.add_argument('-i', '--input', type=str, help='Input corpus (in json format)')
parser.add_argument('-o', '--output', type=str, help='Output directory')
parser.add_argument('-v', '--vocab', type=str, help='Vocabulary File')
parser.add_argument('-lm', '--language-model', type=str, help='Language model (name or path)')
parser.add_argument('--text', type=str, help='Text attribute in json', default='text')

parser.add_argument('--train-dir', type=str, help='Training directory', default='./results')
parser.add_argument('--log-dir', type=str, help='Log directory', default='./logs')
parser.add_argument('--epochs', type=int, help='Number of epochs', default=1)
parser.add_argument('--batch-size', type=int, help='Number of epochs', default=16)
parser.add_argument('--steps', type=int, help='Save steps', default=10_000)
parser.add_argument('--warmup', type=int, help='Warmup steps', default=500)
parser.add_argument('--wd', type=float, help='Weight decay', default=0.01)

args = parser.parse_args()

tmp_corpus = NamedTemporaryFile()
df = pd.read_json(args.input, lines=True)

with open(tmp_corpus.name, 'w') as f:
    f.write("\n".join(i for i in df[args.text].to_list() if i is not None))

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.vocab, max_len=512)

model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path=args.language_model)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=tmp_corpus.name,
    block_size=128,
)

training_args = TrainingArguments(
    output_dir=args.train_dir,          # output directory
    num_train_epochs=args.epochs,              # total number of training epochs
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
    warmup_steps=args.warmup,                # number of warmup steps for learning rate scheduler
    weight_decay=args.wd,               # strength of weight decay
    logging_dir=args.log_dir,            # directory for storing logs
    logging_strategy=IntervalStrategy.EPOCH,
    save_strategy=IntervalStrategy.EPOCH
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model(args.output)
tmp_corpus.close()
