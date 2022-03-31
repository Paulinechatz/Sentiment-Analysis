#!/usr/bin/env python3
import argparse as ap
import emojis
import os
import pandas as pd
import re


def preprocess(x, emoji_text=False, remove_emojis=False, remove_hm_symbol=False, remove_hm_begging=False,
               remove_hm_end=False):
    # Βήμα 1. Αφαίρεση retweets
    x = re.sub('\s*RT\s*@[^:]*:', '', x)

    # Βήμα 2. Μετατροπή σε κείμενο και σε πεζά
    x = x.lower()

    # Βήμα 3. Αφαίρεση τόνων
    x = x.translate({
        ord("ά"): "α", ord("ό"): "ο", ord("έ"): "ε", ord("ί"): "ι", ord("ύ"): "υ", ord("ή"): "η",
        ord("ώ"): "ω", ord("ϊ"): "ι", ord("ΐ"): "ι", ord("ϋ"): "υ"
    })

    tokens = x.split()
    r = []

    for token in tokens:
        # Βήμα 4. Αφαίρεση urls
        if re.match('http[^\s]*(\s|$)', token):
            continue

        # Βήμα 5. Emojis
        if emojis.count(token):
            new_token = ''
            for c in token: # Μετραμε χαρακτηρες γιατι μπορει το token να περιεχει και μη emojs
                if emojis.count(c):
                    if len(new_token) > 0: # Στειλε τους μέχρι στιγμής χαρακτήρες σαν νέο token
                        r.append(new_token)
                        new_token = ''

                    if remove_emojis:
                        continue
                    elif emoji_text:
                        r.append(emojis.decode(c))
                        continue

                new_token += c

            if len(new_token) > 0:
                r.append(new_token)
        else:
            r.append(token)

    # Βήμα 6. Αφαίρεση hashtags και mentions από την αρχή
    if remove_hm_begging:
        for token in r.copy():
            if re.match('#[^\s]*(\s|$)', token) or re.match('@[^\s]*(\s|$)', token):
                r.remove(token)
            else:
                break

    # Βήμα 7. Αφαίρεση hashtags και mentions από το τέλος
    if remove_hm_end:
        for token in reversed(r.copy()):
            if re.match('#[^\s]*(\s|$)', token) or re.match('@[^\s]*(\s|$)', token):
                r.remove(token)
            else:
                break

    text = ' '.join(r)

    # Βήμα 8. Αφαίρεση χαρακτήρων '#' kai '@' από τα hashtags και mentions που έχουν απομείνει
    if remove_hm_symbol:
        text = text.replace('#', '').replace('@', '')

    # TODO: Αφαίρεση αποσιωποιητικών, άνω-κάτω τελείας κλπ
    return text


parser = ap.ArgumentParser(description='Preprocess Palo dataset and corpus')
subparsers = parser.add_subparsers(dest='type')

parser.add_argument('--column', type=str, help='Text column in the dataframe', default='text')

em_group = parser.add_mutually_exclusive_group()
em_group.add_argument('--emoji-text', action='store_true', help='Include emoji text (in english)', default=False)
em_group.add_argument('--remove-emojis', action='store_true', help='Include emoji text (in english)', default=False)

parser.add_argument('--remove-hm-symbol', action='store_true', help='Remove hashtags (#) and mention (@) symbols',
                    default=False)

parser.add_argument('--remove-hm-beggining', action='store_true',
                    help='Remove hashtags and mentions from the beggining of text', default=False)

parser.add_argument('--remove-hm-end', action='store_true',
                    help='Remove hashtags and mentions from the end of text', default=False)


dataset = subparsers.add_parser('dataset', description='Preprocess a dataset (in csv format)')
dataset.add_argument('-i', '--input', type=str, help='Input directory (for dataset dataset)')
dataset.add_argument('-o', '--output', type=str, help='Output directory')

corpus = subparsers.add_parser('corpus', description='Preprocess a corpus (in json format)')
corpus.add_argument('-i', '--input', type=str, help='Input dataset')
corpus.add_argument('-o', '--output', type=str, help='Output dataset')


args = parser.parse_args()

if args.type == 'corpus':
    df = pd.read_json(args.input, lines=True)
    df = df[~df[args.column].isna()]
    df[args.column] = df[args.column].apply(preprocess, emoji_text=args.emoji_text,
                                            remove_emojis=args.remove_emojis,
                                            remove_hm_symbol=args.remove_hm_symbol,
                                            remove_hm_begging=args.remove_hm_beggining,
                                            remove_hm_end=args.remove_hm_end)
    df.to_json(args.output, orient='records', lines=True)
elif args.type == 'dataset':
    for f in os.listdir(args.input):
        if os.path.isfile((os.path.join(args.input, f))):
            print("Processing {}... ".format(os.path.join(args.input, f)), end=' ')
            df = pd.read_csv(os.path.join(args.input, f))
            df[args.column] = df[args.column].apply(preprocess, emoji_text=args.emoji_text,
                                                    remove_emojis=args.remove_emojis,
                                                    remove_hm_symbol=args.remove_hm_symbol,
                                                    remove_hm_begging=args.remove_hm_beggining,
                                                    remove_hm_end=args.remove_hm_end)
            df.to_csv(os.path.join(args.output, f), index=False)
            print(' Done!')
else:
    print('You must specify at least one of \'corpus\' or \'dataset\'')
