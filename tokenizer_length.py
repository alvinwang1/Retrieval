#!/usr/bin/env python3
"""
Inspect how an instruction string is tokenized by a HuggingFace AutoTokenizer.

- Loads a tokenizer (default: ModernBERT)
- Tokenizes the instruction
- Prints:
  * token IDs
  * token strings
  * token → id mapping
  * character span offsets (if supported)

Usage:
  python tokenizer_length.py \
    --model_name answerdotai/ModernBERT-base \
    --instruction "You are an expert legal text evaluator."

"""

import argparse
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_name",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="HF model name or local tokenizer directory",
    )
    ap.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Instruction string to tokenize",
    )
    ap.add_argument(
        "--add_special_tokens",
        action="store_true",
        help="Include special tokens like [CLS]/[SEP]",
    )
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    enc = tokenizer(
        args.instruction,
        add_special_tokens=args.add_special_tokens,
        return_offsets_mapping=True,
    )

    input_ids = enc["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    offsets = enc.get("offset_mapping", None)

    print("=== Instruction ===")
    print(args.instruction)
    print()

    print("=== Tokens ===")
    for i, tok in enumerate(tokens):
        if offsets is not None:
            start, end = offsets[i]
            span = args.instruction[start:end] if start != end else ""
            print(f"{i:>3}: {tok:<20} id={input_ids[i]:>6} span=({start},{end}) '{span}'")
        else:
            print(f"{i:>3}: {tok:<20} id={input_ids[i]:>6}")

    print("\n=== Summary ===")
    print(f"Total tokens: {len(tokens)}")
    print(f"Vocab size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
