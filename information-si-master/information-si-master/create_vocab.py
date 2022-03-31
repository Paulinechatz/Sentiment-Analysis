#!/usr/bin/env python3
from tempfile import TemporaryDirectory
from transformers import RobertaTokenizerFast
from tokenizers import ByteLevelBPETokenizer
import argparse as ap
import pandas as pd


parser = ap.ArgumentParser(description='Create Vocabulary')

parser.add_argument('-i', '--input', type=str, help='Input sentences (in json format)')
parser.add_argument('-o', '--output', type=str, help='Output directory')
parser.add_argument('--text', type=str, help='Text attribute in json', default='text')
parser.add_argument('--size', type=int, help='Input dataset in json format', default=52_000)
parser.add_argument('--min-freq', type=int, help='Minimum word frequency', default=2)

args = parser.parse_args()

# Read the JSON file as pandas DataFrame
df = pd.read_json(args.input, lines=True)

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train_from_iterator(iterator=df[args.text].to_list(), vocab_size=args.size, min_frequency=args.min_freq,
                              special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

tmp_dir = TemporaryDirectory()
tokenizer.save_model(tmp_dir.name)

s = RobertaTokenizerFast.from_pretrained(tmp_dir.name, max_len=512)
s.save_pretrained(args.output)
tmp_dir.cleanup()
