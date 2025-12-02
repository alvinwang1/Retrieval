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
wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key:
    wandb.login(key=wandb_api_key)
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

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = normalize(a)
    b = normalize(b)
    return a @ b.T



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

BATCH_SIZE = 25
async def _gpt_label_single_batch(df_batch: pd.DataFrame, eval_type: str, term: str, system_prompt: str, raw_statute: str) -> pd.DataFrame:
    
    records = [
        {"sentence_id": str(row["sentence_id"]), "text": str(row["text"])}
        for _, row in df_batch.iterrows()
    ]
    user_content = {
        "phrase_of_interest": term,
        "raw_statute": raw_statute,
        "sentences": records,
    }
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_content)},
        ]
    )

    output_text = response.choices[0].message.content
    if output_text.startswith("```"):
        # remove opening fence like ``` or ```json
        output_text = re.sub(r"^```[a-zA-Z]*\n?", "", output_text)
        # remove closing fence ```
        output_text = re.sub(r"\n```$", "", output_text)
    try:
        gpt_labels = json.loads(output_text)
        
        # Handle wrapped responses: {"results": [...]} or {"items": [...]}
        if isinstance(gpt_labels, dict):
            # Try common wrapper keys
            for key in ['results', 'items', 'data', 'labels', 'predictions', 'annotations']:
                if key in gpt_labels and isinstance(gpt_labels[key], list):
                    gpt_labels = gpt_labels[key]
                    break
            # If still a dict and not a list, it might be a single-item response
            if isinstance(gpt_labels, dict) and 'sentence_id' not in gpt_labels:
                print(f"WARNING: Unexpected GPT response format: {list(gpt_labels.keys())}")
                print(f"Response preview: {str(output_text)[:300]}")
                raise ValueError(f"GPT returned dict with keys {list(gpt_labels.keys())}, expected list or wrapped list")
        
        # Convert to DataFrame
        df_gpt = pd.DataFrame(gpt_labels)
        
        # Validate required columns
        if 'sentence_id' not in df_gpt.columns:
            print(f"ERROR: Missing 'sentence_id' column")
            print(f"Available columns: {df_gpt.columns.tolist()}")
            print(f"Full GPT response: {output_text}")
            raise KeyError(f"'sentence_id' column not found in GPT response")
        
        if 'label' not in df_gpt.columns:
            print(f"ERROR: Missing 'label' column")
            print(f"Available columns: {df_gpt.columns.tolist()}")
            print(f"Full GPT response: {output_text}")
            raise KeyError(f"'label' column not found in GPT response")
        
        # Process the data
        df_gpt["sentence_id"] = df_gpt["sentence_id"].astype(str)
        df_gpt["gpt_label_str"] = df_gpt["label"]
        df_gpt["gpt_label_score"] = df_gpt["label"].map(LABEL_MAP)
        
        return df_gpt[["sentence_id", "gpt_label_str", "gpt_label_score"]]
        
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse GPT response as JSON")
        print(f"Response: {output_text}")
        raise
    except Exception as e:
        print(f"ERROR processing GPT response: {e}")
        print(f"Response: {output_text}")
        raise

async def gpt_label_batch(df: pd.DataFrame, eval_type: str, term: str, raw_statute: str) -> pd.DataFrame:
    # Load system prompt once
    if eval_type == "zero_shot":
        prompt_path = "system_prompts/zero_shot.txt"
    elif eval_type == "few_shot":
        prompt_path = "system_prompts/few_shot.txt"  # make sure this exists
    else:
        raise ValueError(f"Unknown eval type for GPT labeling: {eval_type}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    tasks = []
    for start in range(0, len(df), BATCH_SIZE):
        sub = df.iloc[start:start + BATCH_SIZE]
        tasks.append(asyncio.create_task(_gpt_label_single_batch(sub, eval_type, term, system_prompt, raw_statute)))

    # Run all batches concurrently
    batches = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="GPT batches", unit="batch"):
        batch_df = await coro
        batches.append(batch_df)

    df_gpt = pd.concat(batches, ignore_index=True)
    return df_gpt

def log_to_wandb(term: str, eval_type: str, df_st_sorted: pd.DataFrame, ndcg_scores: List[float], label_counts: Dict[str, int], confusion: pd.DataFrame):
    """
    Logs:
      - config: term, type, num_sentences
      - table: all ranked sentences + scores
      - metrics: ndcg@10, ndcg@20, ndcg@40, ndcg@100
    """
    # Initialize run (customize project/name as you like)
    wandb.init(
        project="statutory-term-ndcg-eval",
        config={
            "term": term,
            "type": eval_type,
            "num_sentences": len(df_st_sorted),
            "label_map": LABEL_MAP,
            "label_counts": label_counts,
            "confusion": confusion,
        }
    )

    # Decide which column is the model score
    if eval_type == "naive":
        score_col = "similarity"
    else:
        score_col = "gpt_label_score"

    table = wandb.Table(
        columns=["rank", "sentence_id", "text", "label_score", "pred_score"]
    )
    # Convert confusion matrix to wandb.Table
    desired_order = [
        "no value",
        "potential value",
        "certain value",
        "high value",
    ]
    
    score_to_name = {v: k for (k, v) in LABEL_MAP.items()}

    confusion.index = confusion.index.map(score_to_name)

    sorted_confusion = confusion.reindex(
        index=desired_order,
        columns=desired_order,
    )
    confusion_table = wandb.Table(
        columns=["true_label", "pred no value", "pred potential value", "pred certain value", "pred high value"]
    )
    for true_label, row in sorted_confusion.iterrows():
        confusion_table.add_data(
            true_label,
            int(row.get("no value", 0)),
            int(row.get("potential value", 0)),
            int(row.get("certain value", 0)),
            int(row.get("high value", 0)),
        )
    for rank, (_, row) in enumerate(df_st_sorted.iterrows(), start=1):
        # Some rows might not have label_score if you later change filtering
        label_score = float(row.get("label_score", -1))
        pred_score = float(row.get(score_col, 0.0))
        table.add_data(
            rank,
            str(row["sentence_id"]),
            str(row["text"]),
            label_score,
            pred_score,
        )

    metrics = {
        "ndcg@10": ndcg_scores[0] if len(ndcg_scores) > 0 else None,
        "ndcg@20": ndcg_scores[1] if len(ndcg_scores) > 1 else None,
        "ndcg@40": ndcg_scores[2] if len(ndcg_scores) > 2 else None,
        "ndcg@100": ndcg_scores[3] if len(ndcg_scores) > 3 else None,
    }

    wandb.log({
        "sentences": table,
        "label_counts": label_counts,
        "confusion": confusion_table,
        **metrics,
    })

    wandb.finish()

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", default="sources/independent_economic_value/independent_economic_value-sentence.json", help="Path to sentences json")
    ap.add_argument("--provisions", default="sources/provisions.json", help="Path to provisions json")
    ap.add_argument("--term", default="independent economic value", help="Provision key to use as query")
    ap.add_argument("--type", default="naive", help ="naive | zero_shot | few_shot")
    ap.add_argument("--log", default="false", help="Log to wandb")
    ap.add_argument("--verbose", default="false", help="Verbose output")
    args = ap.parse_args()

    sent_path = Path(args.sentences)
    if not sent_path.exists():
        raise FileNotFoundError(f"File not found: {sent_path}")

    prov_path = Path(args.provisions)
    if not prov_path.exists():
        raise FileNotFoundError(f"File not found: {prov_path}")

    with sent_path.open("r", encoding="utf-8") as f:
        sent_json = json.load(f)
    with prov_path.open("r", encoding="utf-8") as f:
        provisions = json.load(f)

    df = build_records(sent_json)
    if df.empty:
        raise ValueError("No sentences loaded.")

    if args.term not in provisions:
        raise KeyError(f"Term '{args.term}' not found in provisions.json. Available keys: {list(provisions.keys())[:10]} ...")
    raw_statute = provisions[args.term]["raw"]
    analyzed_statute = provisions[args.term]["analyzed"]

    ndcg_scores = []
    df_st_sorted = None

    if args.type == "naive":
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_emb = model.encode([args.term], convert_to_numpy=True)
        sent_embs = model.encode(df["text"].tolist(), batch_size=64, convert_to_numpy=True)

        sims = cosine_sim(sent_embs, query_emb)[:, 0]
        df_st = df.copy()
        df_st["similarity"] = sims
        df_st_sorted = df_st.sort_values("similarity", ascending=False).reset_index(drop=True)

        # NDCG evaluation
        for k in [10, 20, 40, 100]:
            ndcg_scores.append(ndcg_score([df_st_sorted["label_score"].tolist()], [df_st_sorted["similarity"].tolist()], k=k))
    
    if args.type == "zero_shot" or args.type == "few_shot":
        # Get GPT labels for each sentence
        df_gpt = asyncio.run(gpt_label_batch(df, args.type, args.term, raw_statute))
        
        # Merge GPT labels back into the original df (which has text + human labels)
        df_merged = df.merge(df_gpt, on="sentence_id", how="left")

        # Keep only rows with a valid human label (>= 0) and a GPT score
        df_eval = df_merged[df_merged["label_score"] >= 0].copy()
        df_eval = df_eval.dropna(subset=["gpt_label_score"])

        # Ensure numeric types
        df_eval["label_score"] = df_eval["label_score"].astype(float)
        df_eval["gpt_label_score"] = df_eval["gpt_label_score"].astype(float)

        # Sort by GPT's predicted relevance
        y_true = np.array([df_eval["label_score"].values], dtype=float)
        y_score = np.array([df_eval["gpt_label_score"].values], dtype=float)
        
        for i in [10, 20, 40, 100]:
            ndcg_scores.append(ndcg_score(y_true, y_score, k=i))
        df_st_sorted = df_eval.sort_values("gpt_label_score", ascending=False).reset_index(drop=True)



    topk = 10
    if args.type == "naive" and args.verbose == "true":
        for i, row in df_st_sorted.head(topk).iterrows():
            print(f"[{i+1:>2}] sim={row['similarity']:.4f}  rel={row['label_score']:<2}  | {row['text']}")
    elif (args.type == "zero_shot" or args.type == "few_shot") and args.verbose == "true":
        for i, row in df_st_sorted.head(topk).iterrows():
            # use GPT’s predicted score as “sim”
            print(f"[{i+1:>2}] sim={row['gpt_label_score']:.4f}  rel={row['label_score']:<2}  | {row['text']}")

    
    print(f"Method: {args.type} | Term: {args.term}")
    for k, ndcg in zip([10, 20, 40, 100], ndcg_scores):
        print(f"NDCG@{k}: {ndcg}")
    
    label_counts = df_st_sorted["gpt_label_str"].value_counts().to_dict()
    confusion = pd.crosstab(df_st_sorted["label_score"], df_st_sorted["gpt_label_str"])

    if df_st_sorted is not None and len(ndcg_scores) > 0 and args.log == "true":
        log_to_wandb(args.term, args.type, df_st_sorted, ndcg_scores, label_counts, confusion)


if __name__ == "__main__":
    main()