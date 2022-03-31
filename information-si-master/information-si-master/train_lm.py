#!/usr/bin/env python3
from tempfile import NamedTemporaryFile
from transformers import DataCollatorForLanguageModeling, LineByLineTextDataset, RobertaConfig, RobertaForMaskedLM, \
    RobertaTokenizerFast, Trainer, TrainingArguments
from transformers.trainer_utils import IntervalStrategy
import argparse as ap
import pandas as pd

parser = ap.ArgumentParser(description='Train a RoBERTa Language model from scratch')

parser.add_argument('-i', '--input', type=str, help='Input corpus (in json format)')
parser.add_argument('-o', '--output', type=str, help='Output directory')
parser.add_argument('-v', '--vocab', type=str, help='Vocabulary File')
parser.add_argument('--text', type=str, help='Text attribute in json', default='text')

model = parser.add_argument_group('model', description='Model parameters')
model.add_argument('--size', type=int, help='Vocabulary size', default=52_000)
model.add_argument('--heads', type=int, help='Number of attention heads', default=12)
model.add_argument('--hidden-layers', type=int, help='Number of hidden layers', default=6)

train = parser.add_argument_group('train', description='Training parameters')
train.add_argument('--train-dir', type=str, help='Training directory', default='./results')
train.add_argument('--log-dir', type=str, help='Log directory', default='./logs')
train.add_argument('--epochs', type=int, help='Number of epochs', default=1)
train.add_argument('--batch-size', type=int, help='Number of epochs', default=16)
train.add_argument('--steps', type=int, help='Save steps', default=10_000)
train.add_argument('--warmup', type=int, help='Warmup steps', default=500)
train.add_argument('--wd', type=float, help='Weight decay', default=0.01)

args = parser.parse_args()

tmp_corpus = NamedTemporaryFile()
df = pd.read_json(args.input, lines=True)

with open(tmp_corpus.name, 'w') as f:
    f.write("\n".join(i for i in df[args.text].to_list() if i is not None))


config = RobertaConfig(
    vocab_size=args.size,
    max_position_embeddings=514,
    num_attention_heads=args.heads,
    num_hidden_layers=args.hidden_layers,
    type_vocab_size=1,
    num_labels=3,
    label2id={'NEUTRAL': 0, 'POSITIVE': 1, 'NEGATIVE': 2},
    id2label={0: 'NEUTRAL', 1: 'POSITIVE', 2: 'NEGATIVE'}
)

tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_name_or_path=args.vocab, max_len=512)
model = RobertaForMaskedLM(config=config)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=tmp_corpus.name,
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
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
