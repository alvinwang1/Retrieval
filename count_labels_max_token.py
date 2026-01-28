#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# You don't actually need OpenAI / sentence-transformers / wandb for token scanning,
# but keeping your structure is fine.
import os
from openai import AsyncOpenAI  # unused for this script, but kept since you imported it

from transformers import AutoTokenizer

load_dotenv("../.env")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LABEL_MAP = {
    "no value": 0,
    "potential value": 1,
    "certain value": 2,
    "high value": 3,
}


def normalize_entry(term: str, entry: dict) -> Tuple[str, str]:
    if not isinstance(entry, dict):
        raise ValueError(f"Value for key '{term}' must be an object with 'analyzed' and 'raw'.")
    analyzed = entry.get("analyzed", "")
    raw = entry.get("raw", "")
    if not isinstance(analyzed, str):
        analyzed = "" if analyzed is None else str(analyzed)
    if not isinstance(raw, str):
        raw = "" if raw is None else str(raw)
    return analyzed, raw


def build_records(sent_json: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for sid, entry in sent_json.items():
        label_str = (entry.get("label", "") or "").strip().lower()
        rows.append(
            {
                "sentence_id": sid,
                "case_id": entry.get("case_id", ""),
                "opinion_id": entry.get("opinion_id", ""),
                "paragraph_id": entry.get("paragraph_id", ""),
                "position": entry.get("position", -1),
                "text": entry.get("text", ""),
                "label": entry.get("label", ""),
                "label_score": LABEL_MAP.get(label_str, -1),
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provisions", default="sources/provisions.json", help="Path to provisions json")
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
        help="true|false print the max sentence text",
    )
    args = ap.parse_args()

    prov_path = Path(args.provisions)
    if not prov_path.exists():
        raise FileNotFoundError(f"File not found: {prov_path}")

    with prov_path.open("r", encoding="utf-8") as f:
        provisions = json.load(f)

    include_special = (args.include_special_tokens.lower() == "true")
    verbose = (args.verbose.lower() == "true")

    # Initialize tokenizer ONCE for speed
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    total_label_count = {0: 0, 1: 0, 2: 0, 3: 0}

    max_token_len = -1
    max_sentence_text = None
    max_sentence_term = None
    max_sentence_id = None

    token_lengths = []  # for percentiles + average
    total_rows = 0

    for term in provisions.keys():
        label_count = {0: 0, 1: 0, 2: 0, 3: 0}

        term_slug = term.replace(" ", "_")
        sent_path = Path(f"sources/{term_slug}/{term_slug}-sentence.json")
        if not sent_path.exists():
            raise FileNotFoundError(f"File not found: {sent_path}")

        with sent_path.open("r", encoding="utf-8") as f:
            sent_json = json.load(f)

        df = build_records(sent_json)
        if df.empty:
            raise ValueError(f"No sentences loaded for term: {term}")

        for _, row in df.iterrows():
            text = row["text"] or ""

            # Tokenize without truncation to get the true token length
            token_ids = tokenizer(
                text,
                add_special_tokens=include_special,
                truncation=False,
            )["input_ids"]
            token_len = len(token_ids)

            token_lengths.append(token_len)
            total_rows += 1

            if token_len > max_token_len:
                max_token_len = token_len
                max_sentence_text = text
                max_sentence_term = term_slug
                max_sentence_id = row.get("sentence_id")

            label = row["label"]
            if label not in LABEL_MAP:
                raise ValueError(f"Invalid label: {label} (term={term_slug}, sentence_id={row.get('sentence_id')})")

            label_count[LABEL_MAP[label]] += 1

        print(f"Term: {term_slug}, Label Count: {label_count}")

        total_label_count[0] += label_count[0]
        total_label_count[1] += label_count[1]
        total_label_count[2] += label_count[2]
        total_label_count[3] += label_count[3]

    if total_rows == 0:
        raise ValueError("No rows found across all terms.")

    lengths_np = np.array(token_lengths, dtype=float)
    avg_len = float(lengths_np.mean())

    print("\n=== Token length stats ===")
    print(f"Tokenizer: {args.model_name}")
    print(f"Include special tokens: {include_special}")
    print(f"Total rows: {total_rows}")
    print(f"Max token length: {max_token_len}")
    print(f"Average token length: {avg_len:.2f}")
    for p in [50, 75, 90, 95, 99]:
        print(f"P{p}: {np.percentile(lengths_np, p):.0f}")

    print("\n=== Labels ===")
    print(f"Total Label Count: {total_label_count}")

    if verbose:
        print("\n=== Max token-length sentence ===")
        print(f"term: {max_sentence_term}")
        print(f"sentence_id: {max_sentence_id}")
        print(max_sentence_text)


if __name__ == "__main__":
    main()
