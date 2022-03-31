#!/usr/bin/env python3
import argparse as ap
import os
import pandas as pd


parser = ap.ArgumentParser(description='Simple dataset dataset')

parser.add_argument('-i', '--input', type=str, help='Input dataset (in CSV format)')
parser.add_argument('-o', '--output', type=str, help='Output directory')
parser.add_argument('-p', '--pattern', type=str, help='Output files pattern', default='{}.csv.gz')
parser.add_argument('--train', type=float, help='Training ratio', default=0.6)
parser.add_argument('--val', type=float, help='Validation ratio', default=0.2)
parser.add_argument('--text-column', type=str, help='Text column in the dataset', default='text')
parser.add_argument('--sentiment-column', type=str, help='Text column in the dataset', default='sentiment')

args = parser.parse_args()

sent_dict = {'negative': 2, 'positive': 1, 'neutral': 0}

df = pd.read_csv(args.input)
df = df[~df[args.text_column].isna()]
df = df[~df[args.sentiment_column].isna()]
df = df.sample(frac=1)
df[args.sentiment_column] = df[args.sentiment_column].apply(lambda x: int(sent_dict.get(x)))

tr_idx, vl_idx = round(df.shape[0]*args.train), round(df.shape[0]*args.val)

df_train = df[:tr_idx]
df_val = df[tr_idx:tr_idx+vl_idx]
df_test = df[tr_idx+vl_idx:]

df_train.to_csv(os.path.join(args.output, args.pattern.format('train')), index=False)
df_val.to_csv(os.path.join(args.output, args.pattern.format('val')), index=False)
df_test.to_csv(os.path.join(args.output, args.pattern.format('test')), index=False)
