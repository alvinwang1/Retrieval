import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from openai import OpenAI, AsyncOpenAI
import asyncio

import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import re
import wandb

load_dotenv("../.env")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#wandb_api_key = os.getenv("WANDB_API_KEY")
#if wandb_api_key:
#    wandb.login(key=wandb_api_key)
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
        rows.append({
            "sentence_id": sid,
            "case_id": entry.get("case_id", ""),
            "opinion_id": entry.get("opinion_id", ""),
            "paragraph_id": entry.get("paragraph_id", ""),
            "position": entry.get("position", -1),
            "text": entry.get("text", ""),
            "label": entry.get("label", ""),
            "label_score": LABEL_MAP.get(label_str, -1),
        })
    return pd.DataFrame(rows)

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", default="sources/independent_economic_value/independent_economic_value-sentence.json", help="Path to sentences json")
    ap.add_argument("--provisions", default="sources/provisions.json", help="Path to provisions json")
    ap.add_argument("--term", default="independent economic value", help="Provision key to use as query")
    ap.add_argument("--type", default="naive", help ="naive | zero_shot | few_shot_learn | few_shot")
    ap.add_argument("--log", default="false", help="Log to wandb")
    ap.add_argument("--verbose", default="false", help="Verbose output")
    ap.add_argument("--id_check", default="false", help="Run ID check")
    args = ap.parse_args()

    prov_path = Path(args.provisions)
    if not prov_path.exists():
        raise FileNotFoundError(f"File not found: {prov_path}")

    with prov_path.open("r", encoding="utf-8") as f:
        provisions = json.load(f)

    total_label_count = {0: 0, 1: 0, 2: 0, 3: 0}
    for term in provisions.keys():
        label_count = {0: 0, 1: 0, 2: 0, 3: 0}
        term = term.replace(" ", "_")
        sent_path = Path(f"sources/{term}/{term}-sentence.json")
        if not sent_path.exists():
            raise FileNotFoundError(f"File not found: {sent_path}")
        with sent_path.open("r", encoding="utf-8") as f:
            sent_json = json.load(f)
        df = build_records(sent_json)
        if df.empty:
            raise ValueError("No sentences loaded.")
        for idx, row in df.iterrows():
            label = row["label"]
            if label not in LABEL_MAP:
                raise ValueError(f"Invalid label: {label}")
            
            label_count[LABEL_MAP[label]] += 1
        print(f"Term: {term}, Label Count: {label_count}")
        total_label_count[0] += label_count[0]
        total_label_count[1] += label_count[1]
        total_label_count[2] += label_count[2]
        total_label_count[3] += label_count[3]
    print(f"Total Label Count: {total_label_count}")
    print(f"Total rows: {total_label_count[0] + total_label_count[1] + total_label_count[2] + total_label_count[3]}")

   
 

if __name__ == "__main__":
    main()