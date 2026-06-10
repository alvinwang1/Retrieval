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
import random
from sklearn.preprocessing import normalize
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

# 6-fold term distribution for cross-validation
FOLDS: List[List[str]] = [
    [
        "navigation equipment",
        "leadership role in an organization",
        "aural transfer",
        "semiconductor chip product",
        "distributive share of the income",
        "preexisting work",
        "audiovisual work",
    ],
    [
        "nonindustrial use",
        "significant property damage",
        "nonmonetary benefits",
        "basic allowance for subsistence",
        "stored electronically",
        "independent economic value",
        "technological measure",
    ],
    [
        "unduly disrupt the operations",
        "substantial portion of the public",
        "small manufacturer",
        "accommodation trade",
        "standard coin",
        "residential dwelling",
        "common business purpose",
    ],
    [
        "hazardous liquid",
        "fully amortize",
        "security vulnerability",
        "familiar symbol",
        "mechanical recordation",
        "electronic signature",
        "fermented liquor",
    ],
    [
        "hybrid instrument",
        "unreasonably low prices",
        "gas pipeline facility",
        "preemployment testing",
        "final average compensation",
        "identifying particular",
        "useful improvement",
    ],
    [
        "dischargeable consumer debt",
        "cybercrime",
        "digital musical recording",
        "viticultural",
        "dependent on hours worked",
        "essential step",
        "switchblade knife",
    ],
]

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

# Using robust version below at line 736


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
    def safe_int(v):
        try:
            if pd.isna(v): return 0
            return int(v)
        except:
            return 0

    for true_label, row in sorted_confusion.iterrows():
        confusion_table.add_data(
            true_label,
            safe_int(row.get("no value", 0)),
            safe_int(row.get("potential value", 0)),
            safe_int(row.get("certain value", 0)),
            safe_int(row.get("high value", 0)),
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
async def evaluate_single_term(term: str, provisions: dict, system_prompt: str, args) -> Tuple[str, List[float]]:
    """Evaluate a single term and return its NDCG scores."""
    adjusted_term = term.replace(" ", "_")
    sent_path = Path(f"sources/{adjusted_term}/{adjusted_term}-sentence.json")
    
    if not sent_path.exists():
        print(f"WARNING: File not found for term '{term}': {sent_path}")
        return term, []
    
    try:
        with sent_path.open("r", encoding="utf-8") as f:
            sent_json = json.load(f)
        
        df = build_records(sent_json)
        if df.empty:
            print(f"WARNING: No sentences loaded for term '{term}'")
            return term, []
        
        raw_statute = provisions[term]["raw"]
        
        # Get GPT labels
        df_gpt = await gpt_label_batch(df, args.type, term, args.id_check, raw_statute, system_prompt)
        
        # Merge and evaluate
        df_merged = df.merge(df_gpt, on="sentence_id", how="left")
        df_eval = df_merged[df_merged["label_score"] >= 0].copy()
        df_eval = df_eval.dropna(subset=["gpt_label_score"])
        
        if df_eval.empty:
            print(f"WARNING: No valid evaluation data for term '{term}'")
            return term, []
        
        df_eval["label_score"] = df_eval["label_score"].astype(float)
        df_eval["gpt_label_score"] = df_eval["gpt_label_score"].astype(float)
        
        y_true = np.array([df_eval["label_score"].values], dtype=float)
        y_score = np.array([df_eval["gpt_label_score"].values], dtype=float)
        
        ndcg_scores = []
        for k in [10, 20, 40, 100]:
            ndcg_scores.append(ndcg_score(y_true, y_score, k=k))
        
        print(f"✓ Completed term: {term} | NDCG@10: {ndcg_scores[0]:.4f}")
        return term, ndcg_scores
        
    except Exception as e:
        print(f"ERROR processing term '{term}': {e}")
        return term, []


    return term_ndcg


def highlight_sentences_in_paragraph(paragraph_text: str, sentence_items: List[Dict[str, Any]]) -> str:
    """
    paragraph_text: The raw text of the paragraph.
    sentence_items: List of dicts, each must have 'sentence_id' and 'text'.
    Returns: The paragraph text with [[SENTENCE id=...]]...[[/SENTENCE]] markers.
    """
    matches = []  # (start, end, sentence_id)

    for item in sentence_items:
        s_text = item['text']
        s_id = item['sentence_id']

        # 1. Exact match
        try:
            start_idx = paragraph_text.find(s_text)
        except Exception:
            start_idx = -1
            
        if start_idx != -1:
            matches.append({
                'start': start_idx,
                'end': start_idx + len(s_text),
                'id': s_id,
            })
            continue

        # 2. Fuzzy match fallback
        # This is expensive but necessary if text normalization isn't enough
        matcher = difflib.SequenceMatcher(None, paragraph_text, s_text)
        match = matcher.find_longest_match(0, len(paragraph_text), 0, len(s_text))

        # Threshold: matched length should be significant (e.g. > 80% of sentence length)
        # and at least some minimum chars
        if len(s_text) > 0 and (match.size / len(s_text) > 0.8):
             matches.append({
                'start': match.a,
                'end': match.a + match.size,
                'id': s_id,
            })
             continue
        
        # Log failure (optional, maybe too verbose for valid runs)
        # print(f"Warning: Could not find sentence {s_id} in paragraph.")

    # Sort matches by start index
    matches.sort(key=lambda x: x['start'])

    # Resolve overlaps: strictly discard overlaps that start before previous one ends
    final_matches = []
    last_end = 0
    for m in matches:
        if m['start'] >= last_end:
            final_matches.append(m)
            last_end = m['end']
        else:
            # Overlap detected
            pass

    # Construct new string
    result = []
    curr_idx = 0
    for m in final_matches:
        result.append(paragraph_text[curr_idx:m['start']])
        result.append(f"[[SENTENCE id={m['id']}]]")
        result.append(paragraph_text[m['start']:m['end']])
        result.append(f"[[/SENTENCE]]")
        curr_idx = m['end']
    result.append(paragraph_text[curr_idx:])

    return "".join(result)


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    # 0. Pre-process: Strip <think> blocks which confuse the bracket search
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    if '<think>' in text:
        # If still there, it might be unclosed
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)

    # 1. First, try to handle common LLM formatting hallucinations
    # Handle "0. nine" -> "0.9", "0. zero" -> "0.0", etc.
    num_map = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
    }
    for word, digit in num_map.items():
        text = re.sub(rf"0\.\s*{word}", f"0.{digit}", text, flags=re.IGNORECASE)
    
    # Handle leading zeros in numbers (e.g., "00" or "0. 9") which are invalid JSON
    # Replace "certain value": 00 with 0.0
    text = re.sub(r':\s*0(\d+)', r': 0.\1', text)
    # Handle standalone 00 -> 0
    text = re.sub(r':\s*00(?!\.)', r': 0', text)
    
    # 2. Existing JSON extraction logic
    try:
        # Find first [ and last ]
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            return json.loads(json_str)
        
        # If no brackets, maybe it's a raw object with a list inside?
        start_obj = text.find('{')
        end_obj = text.rfind('}')
        if start_obj != -1 and end_obj != -1:
            json_str = text[start_obj:end_obj+1]
            data = json.loads(json_str)
            if isinstance(data, list): return data
            for k in ["results", "output", "ranking", "predictions", "data", "probs"]:
                if k in data and isinstance(data[k], list):
                    return data[k]
        
        return json.loads(text)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        # Show more of the response for debugging truncation/formatting
        print(f"Original Text snippet (first 2000 chars): {text[:2000]}...")
        raise e


async def evaluate_large_context(term: str, sentences_path: Path, paragraphs_path: Path, args) -> Tuple[str, List[float]]:
    try:
        # Load sentences
        with sentences_path.open("r", encoding="utf-8") as f:
            sent_json = json.load(f)
        df = build_records(sent_json)
        
        if df.empty:
            print(f"WARNING: No sentences loaded for term '{term}'")
            return term, []

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

async def run_ranking_engine(args):
    sent_path = Path(args.sentences)
    if not sent_path.exists():
        return args.term, [], f"File not found: {sent_path}"

    prov_path = Path(args.provisions)
    if not prov_path.exists():
        return args.term, [], f"File not found: {prov_path}"

    # For large_context we need paragraph.json too
    if args.type == "large_context":
        para_path = sent_path.parent / f"{args.term.replace(' ', '_')}-paragraph.json"
        if not para_path.exists():
            # Try implicit naming if term doesn't match folder
            fallback = sent_path.parent / f"{sent_path.stem.replace('-sentence', '-paragraph')}.json"
            if fallback.exists():
                para_path = fallback
            else:
                return args.term, [], f"Paragraph file not found: {para_path}"
        await evaluate_large_context(args.term, sent_path, para_path, args)
        return args.term, [], "OK"

    with sent_path.open("r", encoding="utf-8") as f:
        sent_json = json.load(f)
    with prov_path.open("r", encoding="utf-8") as f:
        provisions = json.load(f)

    df = build_records(sent_json)
    if df.empty:
        return args.term, [], "No sentences loaded."

    if args.term not in provisions:
        raise KeyError(f"Term '{args.term}' not found in provisions.json. Available keys: {list(provisions.keys())[:10]} ...")
    raw_statute = provisions[args.term]["raw"]
    analyzed_statute = provisions[args.term]["analyzed"]

    ndcg_scores = []
    df_st_sorted = None
    ndcg_f = get_ndcg_score()

    if args.type == "naive":
        model = get_sentence_transformer()
        query_emb = model.encode([args.term], convert_to_numpy=True)
        sent_embs = model.encode(df["text"].tolist(), batch_size=64, convert_to_numpy=True)

        sims = cosine_sim(sent_embs, query_emb)[:, 0]
        df_st = df.copy()
        df_st["similarity"] = sims
        df_st_sorted = df_st.sort_values("similarity", ascending=False).reset_index(drop=True)

        for k in [10, 20, 40, 100]:
            ndcg_scores.append(ndcg_score([df_st_sorted["label_score"].tolist()], [df_st_sorted["similarity"].tolist()], k=k))
    
    if args.type == "zero_shot" or args.type == "few_shot":
        # Get GPT labels for each sentence
        df_gpt = asyncio.run(gpt_label_batch(df, args.type, args.term, raw_statute))
        
        df_merged = df.merge(df_gpt, on="sentence_id", how="left")
        df_eval = df_merged[df_merged["label_score"] >= 0].copy()
        
        # Ensure gpt_label_score is present
        if "gpt_label_score" not in df_eval or df_eval.empty:
             print(f"Warning: No valid GPT scores for {args.term}")
             return args.term, [0.0, 0.0, 0.0, 0.0], "OK"

        df_eval = df_eval.dropna(subset=["gpt_label_score"])
        if df_eval.empty:
             print(f"Warning: df_eval became empty after dropping NaNs for {args.term}")
             return args.term, [0.0, 0.0, 0.0, 0.0], "OK"

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