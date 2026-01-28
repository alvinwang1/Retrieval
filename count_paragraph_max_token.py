#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv("../.env")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sources_dir",
        default="sources",
        help="Root directory containing per-term subfolders",
    )
    ap.add_argument(
        "--pattern",
        default="*-paragraph.json",
        help="Glob pattern for paragraph files (searched recursively)",
    )
    ap.add_argument(
        "--model_name",
        default="answerdotai/ModernBERT-base",
        help="HF tokenizer/model name used to count tokens",
    )
    ap.add_argument(
        "--include_special_tokens",
        default="true",
        help="true|false include special tokens in token count",
    )
    ap.add_argument(
        "--verbose",
        default="false",
        help="true|false print the max text",
    )
    args = ap.parse_args()

    sources_dir = Path(args.sources_dir)
    if not sources_dir.exists():
        raise FileNotFoundError(f"Directory not found: {sources_dir}")

    # ------------------------------------------------------------------
    # Find all paragraph files RECURSIVELY
    # Layout: sources/<term_slug>/<term_slug>-paragraph.json
    # ------------------------------------------------------------------
    paragraph_files = list(sources_dir.rglob(args.pattern))
    if not paragraph_files:
        raise FileNotFoundError(
            f"No files matching pattern '{args.pattern}' under {sources_dir} (recursive)."
        )

    print(f"Found {len(paragraph_files)} paragraph files")

    include_special = args.include_special_tokens.lower() == "true"
    verbose = args.verbose.lower() == "true"

    # Initialize tokenizer ONCE for speed
    print(f"Loading tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    max_token_len = -1
    max_text = None
    max_file = None
    max_excerpt_id = None

    token_lengths = []
    total_excerpts = 0

    # ------------------------------------------------------------------
    # Process each paragraph file
    # ------------------------------------------------------------------
    for para_file in paragraph_files:
        print(f"Processing: {para_file}...")

        with para_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Unexpected JSON format in {para_file}")

        for excerpt_id, excerpt in data.items():
            if not isinstance(excerpt, dict):
                continue

            text = excerpt.get("text", "")
            if not text:
                continue

            # Tokenize without truncation to get true token length
            token_ids = tokenizer(
                text,
                add_special_tokens=include_special,
                truncation=False,
            )["input_ids"]

            token_len = len(token_ids)
            token_lengths.append(token_len)
            total_excerpts += 1

            if token_len > max_token_len:
                max_token_len = token_len
                max_text = text
                max_file = str(para_file)
                max_excerpt_id = excerpt_id

    if total_excerpts == 0:
        raise ValueError("No excerpts found across all files.")

    lengths_np = np.array(token_lengths, dtype=float)

    print("\n" + "=" * 60)
    print("=== TOKEN LENGTH STATISTICS ===")
    print("=" * 60)
    print(f"Tokenizer: {args.model_name}")
    print(f"Include special tokens: {include_special}")
    print(f"Files processed: {len(paragraph_files)}")
    print(f"Total excerpts: {total_excerpts}")

    print(f"\nMax token length: {max_token_len}")
    print(f"Average token length: {lengths_np.mean():.2f}")
    print(f"Median token length: {np.median(lengths_np):.0f}")

    print("\nPercentiles:")
    for p in [50, 75, 90, 95, 99]:
        print(f"  P{p}: {np.percentile(lengths_np, p):.0f}")

    if verbose:
        print("\n" + "=" * 60)
        print("=== MAX TOKEN-LENGTH EXCERPT ===")
        print("=" * 60)
        print(f"File: {max_file}")
        print(f"Excerpt ID: {max_excerpt_id}")
        print(f"Token length: {max_token_len}")
        print(f"\nText:\n{max_text}")


if __name__ == "__main__":
    main()
