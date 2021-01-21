#!/usr/bin/env python
from transformers import GPT2Tokenizer
from tqdm import tqdm
import pickle
import os


def process_raw_file(filename, tokenizer, max_len, eos_token="<|endoftext|>"):
    print(f"Tokenizing {filename} ...")
    wikitext_lines = open(filename, "rb").readlines()
    tokenized_lines = []
    for line in tqdm(wikitext_lines):
        # strip the line, remove next line character
        line = line.strip()
        if len(line) == 0:
            # skip the line with only \n there
            continue
        line = line.decode("UTF-8")
        line = " ".join([line, eos_token])
        tokenized_line = tokenizer.encode(line, add_prefix_space=True)
        if max_len > 0 and len(tokenized_line) > max_len:
            continue
        tokenized_lines.append(tokenized_line)

    return tokenized_lines


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/path/to/data/wikitext/wikitext-103-raw')
    parser.add_argument('--output-name', default='wikitext103_raw_gpt2bpe')
    parser.add_argument('--max-len', type=int, default=-1)
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenized_wikitext = {}
    for split_name in ["train", "test", "valid"]:
        tokenized_lines = process_raw_file(
            os.path.join(args.data_dir, "wiki.%s.raw" % split_name),
            tokenizer,
            args.max_len
        )
        tokenized_wikitext[split_name] = tokenized_lines

    output_filename = os.path.join(
        args.data_dir,
        '%s%s.pkl' % (args.output_name, '' if args.max_len < 0 else ('_%d' % args.max_len))
    )
    pickle.dump(tokenized_wikitext, open(output_filename, 'wb'))
    print("=== Saved %s" % output_filename)
    print("\ttrain: %d\n\tvalid: %d\n\ttest: %d" % (
        len(tokenized_wikitext['train']),
        len(tokenized_wikitext['valid']),
        len(tokenized_wikitext['test']),
    ))



