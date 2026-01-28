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
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import re
import wandb
import difflib

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

def extract_json_array(text: str):
    """
    Extract the first valid JSON array from text.
    """
    match = re.search(r"\[\s*{.*?}\s*\]", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in GPT output")
    return json.loads(match.group(0))


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
        gpt_labels = extract_json_array(output_text)
        
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
                print(f"Response preview: {output_text}")
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

async def generate_explanation(text: str, label: str, term: str, raw_statute: str) -> str:
    with open("system_prompts/explanation.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
    user_content = {
        "phrase_of_interest": term,
        "raw_statute": raw_statute,
        "sentence": text,
        "annotator_label": label,
    }
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_content)},
        ]
    )
    return response.choices[0].message.content
async def gpt_label_batch(df: pd.DataFrame, eval_type: str, term: str, id_check: bool, raw_statute: str, system_prompt) -> pd.DataFrame:
    # Load system prompt once
    
 
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

async def grab_sentence(provisions, init_label):
    row_counts = [69,1261,139,79,880,71,27,43,135,172,1581,2374,64,2133,210,421,66,359,87,2217,1538,30,18,154,32,204,70,823,3235,84,25,223,452,179,232,366,1646,616,48,508,3867,221]
    target_index = random.randint(0, sum(row_counts) - 1)

    # 3️⃣ map global index → dataset + local offset
    cumulative = 0
    chosen_dataset = None
    local_offset = None
    p = list(provisions.keys())

    for name, count in zip(p, row_counts):
        if cumulative + count > target_index:
            chosen_dataset = name
            local_offset = target_index - cumulative
            break
        cumulative += count


    # 4️⃣ load only that dataset and grab the row

    term = chosen_dataset.replace(" ", "_")
    sent_path = Path(f"sources/{term}/{term}-sentence.json")

    ds_name = chosen_dataset.replace(" ", "_")
    sent_path = Path(f"sources/{ds_name}/{ds_name}-sentence.json")

    with sent_path.open("r", encoding="utf-8") as f:
        sent_json = json.load(f)

    chosen_df = build_records(sent_json)

   
    # select by **row position**, not column label
    row = chosen_df.iloc[local_offset]
    label = row["label"]
    raw_statute = provisions[chosen_dataset]["raw"]
    if(LABEL_MAP[label] != init_label):
        return None
    explanation = await generate_explanation(row['text'], label, chosen_dataset, raw_statute)
    generated_examples = f"\nPhrase of Interest: {chosen_dataset}\nRaw Statute: {raw_statute}\nSentence: {row['text']}\nAnnotator Label: {label}\nExplanation: {explanation}\n"
    print(generated_examples)
    return generated_examples

async def retrieve_system_prompt(term, eval_type, df, raw_statute, id_check, provisions):
    final_rules_path = None
    if eval_type == "zero_shot" or eval_type == "zero_shot_full":
        prompt_path = "system_prompts/zero_shot.txt"
    elif eval_type == "few_shot":
        prompt_path = "system_prompts/few_shot.txt"  # make sure this exists
    elif eval_type == "few_shot_learn" or eval_type == "few_shot_learn_full":
        prompt_path = "system_prompts/few_shot_learn.txt"  # make sure this exists
        final_rules_path = "system_prompts/final_rules.txt"
    else:
        raise ValueError(f"Unknown eval type for GPT labeling: {eval_type}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    if eval_type == "few_shot_learn_full":
        generated_examples = ""
        # 1️⃣ your dataset names and row counts (same order)

        # 2️⃣ pick random global row index
        # grab total of 8 total in order
        counter = 0
        while(counter != 2):
            example = await grab_sentence(provisions, 0)
            if example is not None:
                generated_examples += example
                counter += 1
        while(counter != 4):
            example = await grab_sentence(provisions, 1)
            if example is not None:
                generated_examples += example
                counter += 1
        while(counter != 6):
            example = await grab_sentence(provisions, 2)
            if example is not None:
                generated_examples += example
                counter += 1
        while(counter != 8):
            example = await grab_sentence(provisions, 3)
            if example is not None:
                generated_examples += example
                counter += 1
        final_rules_prompt = ""
        with open(final_rules_path, "r", encoding="utf-8") as f:
            final_rules_prompt = f.read()
        final_rules_prompt = generated_examples + final_rules_prompt
        system_prompt += "\n" + final_rules_prompt

    if final_rules_path is not None:
        
        generated_examples = ""
        # find 2 examples of each label in df and add to final_rules_prompt
        for label in LABEL_MAP.keys():
            mask = df["label_score"] == LABEL_MAP[label]
            examples = df[mask].sample(n=2, random_state=None)
            df = df.drop(examples.index)
          

            for idx, row in examples.iterrows():
                explanation = await generate_explanation(row['text'], label, term, raw_statute)
                if id_check == "true":
                    print(idx, row["text"])
                    print(f"Explanation: {explanation}\n")

                generated_examples += f"\nPhrase of Interest: {term}\nRaw Statute: {raw_statute}\nSentence: {row['text']}\nAnnotator Label: {label}\nExplanation: {explanation}\n"
        final_rules_prompt = ""
        with open(final_rules_path, "r", encoding="utf-8") as f:
            final_rules_prompt = f.read()
        final_rules_prompt = generated_examples + final_rules_prompt
        system_prompt += "\n" + final_rules_prompt
    return system_prompt
    

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


async def evaluate_large_context(term: str, sentences_path: Path, paragraphs_path: Path, args) -> Tuple[str, List[float]]:
    try:
        # Load sentences
        with sentences_path.open("r", encoding="utf-8") as f:
            sent_json = json.load(f)
        df = build_records(sent_json)
        
        if df.empty:
            print(f"WARNING: No sentences loaded for term '{term}'")
            return term, []

        # Load paragraphs
        with paragraphs_path.open("r", encoding="utf-8") as f:
            para_json = json.load(f)
        
        # Initial stats
        print(f"Loaded {len(df)} sentences and {len(para_json)} paragraphs for term '{term}'")

        # Group sentences by paragraph_id
        # We need the sentence dictionary items from the original JSON or reconstruct them from DF
        # Reconstructing from DF is fine.
        
        sentences_by_para = {}
        all_candidate_ids = []
        
        for _, row in df.iterrows():
            pid = str(row['paragraph_id'])
            if pid not in sentences_by_para:
                sentences_by_para[pid] = []
            
            sentences_by_para[pid].append({
                "sentence_id": str(row['sentence_id']),
                "text": str(row['text'])
            })
            all_candidate_ids.append(str(row['sentence_id']))

        # Create Short ID Mapping
        uuid_to_short = {uid: f"S{i}" for i, uid in enumerate(all_candidate_ids)}
        short_to_uuid = {f"S{i}": uid for i, uid in enumerate(all_candidate_ids)}

        # Prepare Context
        # We will iterate over all paragraphs in para_json (assuming order matters or we just take them all)
        # The schema seems to use UUID keys for paragraphs
        
        # Let's try to maintain some stable order if possible, otherwise just dict order
        # The prompt needs simply all paragraphs.
        
        context_parts = []
        highlighted_count = 0
        total_sentences_count = len(df)
        
        for pid, p_data in para_json.items():
            text = p_data.get("text", "")
            
            # Highlight known sentences in this paragraph
            s_list = sentences_by_para.get(pid, [])
            if s_list:
                # Use Short IDs for highlighting
                s_list_short = []
                for item in s_list:
                    uid = item['sentence_id']
                    sid = uuid_to_short.get(uid, uid) # fallback if needed
                    s_list_short.append({'sentence_id': sid, 'text': item['text']})
                    
                highlighted_text = highlight_sentences_in_paragraph(text, s_list_short)
                # Count how many markers we inserted
                highlighted_count += highlighted_text.count("[[SENTENCE")
            else:
                highlighted_text = text
                
            context_parts.append(f"Paragraph {pid}:\n{highlighted_text}")

        full_context = "\n\n".join(context_parts)
        
        print(f"Highlighting stats: Highlighted {highlighted_count} out of {total_sentences_count} candidates.")

        # Construct System Prompt
        system_instructions = (
            "You are a legal expert assisting with relevant case law retrieval.\n"
            "Your task is to rank the likelihood that specific sentences are relevant to a given legal term.\n"
            "You will be provided with:\n"
            "1. The legal phrase of interest.\n"
            "2. A large context containing multiple paragraphs from legal documents. "
            "Some sentences in these paragraphs are highlighted with tags [[SENTENCE id=...]] ... [[/SENTENCE]].\n"
            "3. A list of Candidate Sentence IDs that you must rank.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- You must read the Full Context and evaluate the relevance of each Candidate Sentence to the phrase of interest.\n"
            "- Rank ALL Candidate Sentence IDs from MOST relevant to LEAST relevant.\n"
            "- Output strictly a JSON object with a single key 'ranked_sentence_ids' containing the list of IDs.\n"
            "- You MUST include EVERY single Candidate ID in your output list. Do not omit any.\n"
            "- If a sentence is not relevant, place it at the bottom of the list, but it MUST be included.\n"
            "- Do not include any IDs that are not in the Candidates list.\n"
            "- Double check that the count of IDs in your output matches the count of Candidates provided.\n"
        )
        
        # --- Context Truncation Logic ---
        # Heuristic: 1 token approx 3 chars. 
        # OpenAI Limit: 128k tokens. 
        # Target: ~80k tokens -> ~250,000 chars (extremely conservative).
        MAX_CHARS = 250_000 
        
        # Estimate static parts
        # Note: Using short IDs for candidates now
        all_short_ids = [uuid_to_short[uid] for uid in all_candidate_ids]
        static_chars = len(system_instructions) + len(term) + len(str(all_short_ids))
        remaining_chars = MAX_CHARS - static_chars
        
        # Initialize user_content dictionary
        user_content = {
            "phrase_of_interest": term,
            "candidates": all_short_ids,
            "full_context": full_context
        }
        
        if len(full_context) > remaining_chars:
            print(f"Warning: Context length ({len(full_context)} chars) exceeds budget ({remaining_chars} chars). Truncating...")
            
            # Re-assemble context with prioritization
            # Priority 1: Paragraphs with highlights
            # Priority 2: Other paragraphs
            
            highlighted_paras = []
            background_paras = []
            
            for pid, p_data in para_json.items():
                text = p_data.get("text", "")
                s_list = sentences_by_para.get(pid, [])
                
                if s_list:
                    # Re-highlight with Short IDs
                    s_list_short = []
                    for item in s_list:
                        uid = item['sentence_id']
                        sid = uuid_to_short.get(uid, uid)
                        s_list_short.append({'sentence_id': sid, 'text': item['text']})
                    
                    hl_text = highlight_sentences_in_paragraph(text, s_list_short)
                    highlighted_paras.append(f"Paragraph {pid}:\n{hl_text}")
                else:
                    background_paras.append(f"Paragraph {pid}:\n{text}")
            
            # Fill budget
            final_parts = []
            current_len = 0
            
            # Add all highlighted paragraphs first
            for p in highlighted_paras:
                if current_len + len(p) < remaining_chars:
                    final_parts.append(p)
                    current_len += len(p) + 2 # +2 for newline
                else:
                    # This is bad, even highlights don't fit. 
                    # Truncate the paragraph itself? Or just stop.
                    # Stopping is safer to avoid breaking markup.
                    print(f"Critical: Even highlighted paragraphs exceed limit! Truncating at {len(final_parts)} highlighted paragraphs.")
                    break
            
            # Fill rest with background if space remains
            if current_len < remaining_chars:
                for p in background_paras:
                    if current_len + len(p) < remaining_chars:
                        final_parts.append(p)
                        current_len += len(p) + 2
                    else:
                        break # No more space
            
            full_context = "\n\n".join(final_parts)
            print(f"Truncated context length: {len(full_context)} chars.")
            
        # Call LLM loop
        # Iterative Ranking: If too many sentences are missing, call again with remaining candidates.
        
        final_ranked_ids = []
        candidates_to_rank = all_candidate_ids[:] # make a copy
        
        iteration = 0
        MAX_ITERATIONS = 5
        
        # Switching to gpt-4o as default for potentially better long-context handling
        model_name = args.model if hasattr(args, 'model') and args.model else "gpt-4o"

        while candidates_to_rank and iteration < MAX_ITERATIONS:
            iteration += 1
            if args.verbose == "true" or iteration > 1:
                print(f"--- Iteration {iteration}: Ranking {len(candidates_to_rank)} candidates ---")
            
            # Update prompts with current candidates
            # Estimate static parts (recalc since candidates changed)
            current_short_candidates = [uuid_to_short[uid] for uid in candidates_to_rank]
            static_chars = len(system_instructions) + len(term) + len(str(current_short_candidates))
            remaining_chars = MAX_CHARS - static_chars
            
            user_content = {
                "phrase_of_interest": term,
                "candidates": current_short_candidates,
                "full_context": full_context 
            }
            
            # Check length again just in case candidates list grew? Unlikely, it shrinks.
            if len(full_context) > remaining_chars:
                pass # Already truncated mostly

            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": json.dumps(user_content)},
                ]
            )
            
            output_text = response.choices[0].message.content
            if args.verbose == "true":
                 print(f"--- Response Snippet ---\n{output_text[:500]}...\n----------------")

            # Robust parsing: Extract Short IDs (S0, S1, etc)
            # Pattern: simple S\d+ but we want to be reasonably safe against 'S123' in random text?
            # The model is outputting a JSON list of IDs.
            # Let's simple match S\d+ and filter by what is valid.
            short_id_pattern = r'S\d+' 
            batch_ranked_shorts = re.findall(short_id_pattern, output_text)
            
            if not batch_ranked_shorts:
                 print("Warning: No IDs found in response.")
            else:
                 for sid in batch_ranked_shorts:
                     # Map Short -> UUID
                     if sid in short_to_uuid:
                         uid = short_to_uuid[sid]
                         if uid in candidates_to_rank:
                             final_ranked_ids.append(uid)
                             candidates_to_rank.remove(uid)

        ranked_ids = final_ranked_ids

        # Post-processing ranking
        # Map IDs to ranks. 
        # Missing IDs? Append them at end.
        
        final_ranking = [rid for rid in ranked_ids if rid in all_candidate_ids]
        seen = set(final_ranking)
        missing = [rid for rid in all_candidate_ids if rid not in seen]
        
        # Log dropped stats
        if missing:
            print(f"Stats: {len(missing)} sentences were missing from LLM response and appended to end.")
        else:
            print(f"Stats: 0 sentences missing (100% ranked by LLM).")
            
        final_ranking.extend(missing)
        
        # Assign scores: higher score = better rank. 
        # Let's say score = len(items) - rank_index
        
        score_map = {rid: (len(final_ranking) - idx) for idx, rid in enumerate(final_ranking)}
        
        # Build results DataFrame
        df["gpt_label_score"] = df["sentence_id"].map(lambda x: score_map.get(str(x), 0))
        
        # Filter for eval
        df_eval = df[df["label_score"] >= 0].copy()
        
        y_true = np.array([df_eval["label_score"].values], dtype=float)
        y_score = np.array([df_eval["gpt_label_score"].values], dtype=float)
        
        scores = []
        for k in [10, 20, 40, 100]:
            scores.append(ndcg_score(y_true, y_score, k=k))
            
        print(f"✓ Completed large_context for term: {term} | NDCG@10: {scores[0]:.4f} | NDCG@20: {scores[1]:.4f} | NDCG@40: {scores[2]:.4f} | NDCG@100: {scores[3]:.4f}")
        return term, scores

    except Exception as e:
        print(f"ERROR in large_context for term '{term}': {e}")
        # import traceback
        # traceback.print_exc()
        return term, []


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", default="sources/independent_economic_value/independent_economic_value-sentence.json", help="Path to sentences json")
    ap.add_argument("--provisions", default="sources/provisions.json", help="Path to provisions json")
    ap.add_argument("--term", default="independent economic value", help="Provision key to use as query")
    ap.add_argument("--type", default="naive", help ="naive | zero_shot | few_shot_learn | few_shot | large_context")
    ap.add_argument("--model", default="gpt-4o", help="Model to use (default: gpt-4o)")
    ap.add_argument("--log", default="false", help="Log to wandb")
    ap.add_argument("--verbose", default="false", help="Verbose output")
    ap.add_argument("--id_check", default="false", help="Run ID check")
    args = ap.parse_args()

    sent_path = Path(args.sentences)
    if not sent_path.exists():
        raise FileNotFoundError(f"File not found: {sent_path}")

    prov_path = Path(args.provisions)
    if not prov_path.exists():
        raise FileNotFoundError(f"File not found: {prov_path}")

    # For large_context we need paragraph.json too
    # Assuming it's in the same directory as sentences
    if args.type == "large_context":
        para_path = sent_path.parent / f"{args.term.replace(' ', '_')}-paragraph.json"
        if not para_path.exists():
            # Try implicit naming if term doesn't match folder
            fallback = sent_path.parent / f"{sent_path.stem.replace('-sentence', '-paragraph')}.json"
            if fallback.exists():
                para_path = fallback
            else:
                 raise FileNotFoundError(f"Paragraph file not found: {para_path}")
                 
        asyncio.run(evaluate_large_context(args.term, sent_path, para_path, args))
        return

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
    if args.type == "zero_shot" or args.type == "few_shot" or args.type == "few_shot_learn":
        # Get GPT labels for each sentence

        system_prompt = asyncio.run(
            retrieve_system_prompt(args.term, args.type, df, raw_statute, args.id_check, provisions)
        )
        print(system_prompt)
        df_gpt = asyncio.run(gpt_label_batch(df, args.type, args.term, args.id_check, raw_statute, system_prompt))
        
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
    term_ndcg = {}
    if args.type == "few_shot_learn_full" or args.type == "zero_shot_full":
        # Generate system prompt once (with examples from all provisions)
        system_prompt = asyncio.run(
            retrieve_system_prompt(args.term, args.type, df, raw_statute, args.id_check, provisions)
        )
        
        print("System prompt generated. Starting parallel evaluation...")
        
        # Evaluate all terms in parallel
        term_ndcg = asyncio.run(evaluate_all_terms_parallel(provisions, system_prompt, args))
        term_ndcg = dict(sorted(term_ndcg.items()))

        # Calculate and display results
        all_values = [v for scores in term_ndcg.values() for v in scores]
        
        if all_values:
            avg_ndcg_10 = np.mean([scores[0] for scores in term_ndcg.values() if len(scores) > 0])
            avg_ndcg_20 = np.mean([scores[1] for scores in term_ndcg.values() if len(scores) > 1])
            avg_ndcg_40 = np.mean([scores[2] for scores in term_ndcg.values() if len(scores) > 2])
            avg_ndcg_100 = np.mean([scores[3] for scores in term_ndcg.values() if len(scores) > 3])
            
            print(f"\n{'='*60}")
            print(f"Results for {len(term_ndcg)} terms:")
            print(f"{'='*60}")
            print(f"Average NDCG@10:  {avg_ndcg_10:.4f}")
            print(f"Average NDCG@20:  {avg_ndcg_20:.4f}")
            print(f"Average NDCG@40:  {avg_ndcg_40:.4f}")
            print(f"Average NDCG@100: {avg_ndcg_100:.4f}")
            print(f"{'='*60}\n")
            
            # Log to wandb if requested
            if args.log == "true":
                wandb.init(
                    project="statutory-term-ndcg-eval",
                    config={
                        "type": args.type,
                        "num_terms": len(term_ndcg),
                        "system_prompt": system_prompt[:500],  # First 500 chars
                    }
                )
                
                # Log individual term results
                for term, scores in term_ndcg.items():
                    wandb.log({
                        f"{term}/ndcg@10": scores[0],
                        f"{term}/ndcg@20": scores[1],
                        f"{term}/ndcg@40": scores[2],
                        f"{term}/ndcg@100": scores[3],
                    })
                
                # Log averages
                wandb.log({
                    "avg_ndcg@10": avg_ndcg_10,
                    "avg_ndcg@20": avg_ndcg_20,
                    "avg_ndcg@40": avg_ndcg_40,
                    "avg_ndcg@100": avg_ndcg_100,
                })
                wandb.log({"system_prompt": system_prompt})

                wandb.finish()
        
        return


    topk = 10
    if args.type == "naive" and args.verbose == "true" and df_st_sorted is not None:
        for i, row in df_st_sorted.head(topk).iterrows():
            print(f"[{i+1:>2}] sim={row['similarity']:.4f}  rel={row['label_score']:<2}  | {row['text']}")
    elif (args.type == "zero_shot" or args.type == "few_shot" or args.type == "few_shot_learn") and args.verbose == "true" and df_st_sorted is not None:
        for i, row in df_st_sorted.head(topk).iterrows():
            # use GPT’s predicted score as “sim”
            print(f"[{i+1:>2}] sim={row['gpt_label_score']:.4f}  rel={row['label_score']:<2}  | {row['text']}")

    
    print(f"Method: {args.type} | Term: {args.term}")
    if ndcg_scores:
        for k, ndcg in zip([10, 20, 40, 100], ndcg_scores):
            print(f"NDCG@{k}: {ndcg}")
    
    if df_st_sorted is not None:
        label_counts = df_st_sorted["gpt_label_str"].value_counts().to_dict() if "gpt_label_str" in df_st_sorted else {}
        confusion = pd.crosstab(df_st_sorted["label_score"], df_st_sorted["gpt_label_str"]) if "gpt_label_str" in df_st_sorted else None
    
        if len(ndcg_scores) > 0 and args.log == "true":
            log_to_wandb(args.term, args.type, df_st_sorted, ndcg_scores, label_counts, confusion)
            
    if args.type == "few_shot_learn_full":
        all_values = [v for scores in term_ndcg.values() for v in scores]
        avg_ndcg = sum(all_values) / len(all_values)
        log_to_wandb(term_ndcg, avg_ndcg, system_prompt)


if __name__ == "__main__":
    main()