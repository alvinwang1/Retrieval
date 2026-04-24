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
import difflib

# Lazy imports for performance
_sentence_transformer_model = None

def get_sentence_transformer():
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_transformer_model

def get_ndcg_score():
    from sklearn.metrics import ndcg_score
    return ndcg_score

load_dotenv("../.env")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key:
    wandb.login(key=wandb_api_key)

groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = None
if groq_api_key:
    groq_client = AsyncOpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
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

BATCH_SIZE = 5
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
        temperature=0.0,
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
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_content)},
        ]
    )
    return response.choices[0].message.content

async def _gpt_probabilities_single_batch(df_batch: pd.DataFrame, term: str, system_prompt: str, raw_statute: str, model: str = "gpt-4o-mini") -> pd.DataFrame:
    BATCH_SIZE_INTERNAL = 5  # Ensure we don't exceed local batch logic

    records = [
        {"sentence_id": str(row["sentence_id"]), "text": str(row["text"])}
        for _, row in df_batch.iterrows()
    ]
    user_content = {
        "phrase_of_interest": term,
        "raw_statute": raw_statute,
        "sentences": records,
    }
    
    # Global cleanup of labeling terminology to break LLM persistence
    cleaned_prompt = system_prompt.replace('"label"', '"probs"')
    cleaned_prompt = cleaned_prompt.replace(' label ', ' score distribution ')
    cleaned_prompt = cleaned_prompt.replace(' labeled ', ' scored ')
    
    # Determine which client to use
    active_client = client
    if groq_client and ("gpt-oss" in model or "llama" in model.lower() or "mixtral" in model.lower() or "qwen" in model.lower()):
        active_client = groq_client
        
    try:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = await active_client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    max_tokens=8192,
                    messages=[
                        {"role": "system", "content": cleaned_prompt + "\n\nIMPORTANT: Return ONLY the JSON array. Do NOT use <think> tags. Keep your 'reasoning' field under 15 words to ensure the entire batch is processed without truncation."},
                        {"role": "user", "content": f"Phrase: {term}\nStatute: {raw_statute}\nSentences to evaluate: {json.dumps(records)}"}
                    ]
                )
                output_text = response.choices[0].message.content
                
                # Try to parse the JSON immediately within the retry loop
                try:
                    gpt_data = extract_json_array(output_text)
                    # If gpt_data is a dict (wrapped), try to find the list inside
                    if isinstance(gpt_data, dict):
                        for key in ["results", "output", "ranking", "predictions", "data"]:
                            if key in gpt_data and isinstance(gpt_data[key], list):
                                gpt_data = gpt_data[key]
                                break
                        if isinstance(gpt_data, dict): # Still a dict?
                            for v in gpt_data.values():
                                if isinstance(v, list):
                                    gpt_data = v
                                    break
                    break # Successfully parsed!
                except Exception as parse_error:
                    if attempt < max_retries - 1:
                        print(f"Parse error for {term}, retrying... ({str(parse_error)[:50]})")
                        await asyncio.sleep(2)
                        continue
                    else:
                        raise parse_error

            except Exception as e:
                error_str = str(e).lower()
                if ("rate_limit_exceeded" in error_str or "json_validate_failed" in error_str) and attempt < max_retries - 1:
                    wait_time = 3.0  # Slightly longer default wait
                    # Try to extract wait time from error message if possible
                    match = re.search(r"try again in ([\d.]+)s", str(e))
                    if match:
                        wait_time = float(match.group(1)) + 1.0
                    elif "ms" in str(e):
                        match_ms = re.search(r"try again in ([\d.]+)ms", str(e))
                        if match_ms:
                            wait_time = (float(match_ms.group(1)) / 1000.0) + 0.5
                    
                    print(f"Transient error ({'RateLimit' if 'rate_limit' in error_str else 'JSON'}) for {term}, retrying in {wait_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"FAILED on {term}: {str(e)}")
                    raise e

        # Proceed to process gpt_data (it's guaranteed to be safely parsed if we reached here)
        results = []
        for item in gpt_data:
            try:
                sid = str(item.get("sentence_id", item.get("id", "")))
                if not sid:
                    continue
                
                # Find the scoring data (could be 'probs' or 'label')
                scoring_data = item.get("probs", item.get("label"))
                
                if isinstance(scoring_data, dict):
                    probs = scoring_data
                elif isinstance(scoring_data, str):
                    # Map string label to 1.0 probability for that class
                    label_val = scoring_data.lower()
                    probs = {k: (1.0 if k == label_val else 0.0) for k in LABEL_MAP.keys()}
                else:
                    # Fallback if no scoring data
                    probs = {"no value": 1.0}

                # EV calculation (Expected Value)
                # Ensure we handle non-numeric values that might have slipped through
                def safe_float(v):
                    try: return float(v)
                    except: return 0.0

                ev = (
                    safe_float(probs.get("no value", 0.0)) * 0 +
                    safe_float(probs.get("potential value", 0.0)) * 1 +
                    safe_float(probs.get("certain value", 0.0)) * 2 +
                    safe_float(probs.get("high value", 0.0)) * 3
                )
                results.append({
                    "sentence_id": sid,
                    "gpt_label_score": ev,
                    "gpt_label_str": "distribution"
                })
            except Exception as item_err:
                print(f"Skipping malformed sentence item for {term}: {item_err}")
                continue
        return pd.DataFrame(results)
    except Exception as e:
        print(f"ERROR processing GPT probabilities: {e}")
        # At this point output_text may or may not be defined depending on where it failed
        if 'output_text' in locals():
            print(f"Response: {output_text}")
        raise e

async def gpt_label_batch(df: pd.DataFrame, eval_type: str, term: str, id_check: bool, raw_statute: str, system_prompt, verbose: bool = False, model: str = "gpt-4o-mini") -> pd.DataFrame:
    # Limit total concurrent requests to the API to prevent spamming/rate-limits
    semaphore = asyncio.Semaphore(3)
    
    async def sem_task(sub_df):
        async with semaphore:
            if eval_type == "probabilities":
                return await _gpt_probabilities_single_batch(sub_df, term, system_prompt, raw_statute, model=model)
            else:
                return await _gpt_label_single_batch(sub_df, eval_type, term, system_prompt, raw_statute)

    tasks = []
    for start in range(0, len(df), BATCH_SIZE):
        sub = df.iloc[start:start + BATCH_SIZE]
        tasks.append(asyncio.create_task(sem_task(sub)))
    
    results = await asyncio.gather(*tasks)
    return pd.concat(results, ignore_index=True)

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
    # Use temperature 0.0 for explanations too
    explanation = await generate_explanation(row['text'], label, chosen_dataset, raw_statute)
    generated_examples = f"\nPhrase of Interest: {chosen_dataset}\nRaw Statute: {raw_statute}\nSentence: {row['text']}\nAnnotator Label: {label}\nExplanation: {explanation}\n"
    return generated_examples

async def retrieve_system_prompt(term, eval_type, df, raw_statute, id_check, provisions, prompt_path_override=None):
    final_rules_path = None
    if prompt_path_override:
        prompt_path = prompt_path_override
    elif eval_type == "zero_shot" or eval_type == "zero_shot_full" or eval_type == "probabilities":
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
    
    if eval_type == "probabilities" and "### MANDATORY OUTPUT FORMAT" not in system_prompt:
        prob_format = "✅ OUTPUT FORMAT (STRICT) - PROBABILITIES MODE\n\n" \
                     "Return ONLY a JSON object with a \"results\" key. Each record must include a \"reasoning\" field and a \"probs\" field with probabilities for EACH class:\n\n" \
                     "{\n" \
                     " \"results\": [\n" \
                     "  {\n" \
                     "    \"sentence_id\": \"...\",\n" \
                     "    \"reasoning\": \"...\",\n" \
                     "    \"probs\": {\"no value\": 0.1, \"potential value\": 0.2, \"certain value\": 0.3, \"high value\": 0.4}\n" \
                     "  }\n" \
                     " ]\n" \
                     "}"
        
        # Extra cleaning: replace labeling terminology with probability terminology
        system_prompt = system_prompt.replace("MUST be labeled", "MUST be assigned a probability for")
        system_prompt = system_prompt.replace("labeled \"no value\"", "assigned a probability for \"no value\"")
        
        # Regex patterns for different prompt versions
        patterns = [
            r"✅ OUTPUT FORMAT \(STRICT\).*?(?=✅)",
            r"Return ONLY valid JSON in the following exact format:.*?Return only the JSON array\.?",
            r"Assign exactly ONE label per sentence from:.*?\[.*?\]",
            r"Assign exactly ONE label per sentence from:.*?{.*?}",
            r"Assign exactly ONE label per sentence from:.*?\n",
            r"Return ONLY valid JSON.*?Return only the JSON array\.?",
            r"\[\s*{\s*\"sentence_id\".*?\"label\".*?}\s*\]"
        ]
        
        import re
        for pattern in patterns:
            system_prompt = re.sub(pattern, "", system_prompt, flags=re.DOTALL)
        
        # Ensure the probability format is at the top AND bottom clearly
        instr = f"\n\n### MANDATORY OUTPUT FORMAT (PROBABILITIES MODE)\n{prob_format}\n"
        system_prompt = instr + system_prompt + instr
        
        # Final cleanup for any leftover label refs
        system_prompt = system_prompt.replace('"label"', '"probs"')
    if eval_type == "few_shot" or eval_type == "few_shot_learn_full":
        # Identify Fold for target term
        target_fold_idx = -1
        for i, fold in enumerate(FOLDS):
            if term in fold:
                target_fold_idx = i
                break
        
        other_fold_terms = []
        for i, fold in enumerate(FOLDS):
            if i != target_fold_idx:
                other_fold_terms.extend(fold)
        
        # Build dictionary of terms -> provisions for other folds
        other_provisions = {t: provisions[t] for t in other_fold_terms if t in provisions}
        
        generated_examples = ""
        # grab total of 8 total (2 per class)
        for label_val in [0, 1, 2, 3]:
            count = 0
            while count < 2:
                example = await grab_sentence(other_provisions, label_val)
                if example is not None:
                    generated_examples += example
                    count += 1
        
        if eval_type == "few_shot_learn_full":
            final_rules_path = "system_prompts/final_rules.txt"
            final_rules_prompt = ""
            with open(final_rules_path, "r", encoding="utf-8") as f:
                final_rules_prompt = f.read()
            final_rules_prompt = generated_examples + final_rules_prompt
            system_prompt += "\n" + final_rules_prompt
        else:
            # Just append examples for regular few-shot
            system_prompt += "\nExpert Examples for Guidance:\n" + generated_examples

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
        # Limits: gpt-4o-mini: 250k, Others (gpt-4o/gpt-5.2): 900k
        MAX_CHARS = 250_000 if args.model == "gpt-4o-mini" else 900_000
        
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
    ap.add_argument("--type", default="naive", help ="naive | zero_shot | few_shot | few_shot_learn | probabilities | large_context")
    ap.add_argument("--model", default="gpt-4o-mini", help="Model to use (default: gpt-4o-mini)")
    ap.add_argument("--log", default="false", help="Log to wandb")
    ap.add_argument("--verbose", default="false", help="Verbose output")
    ap.add_argument("--id_check", default="false", help="Run ID check")
    ap.add_argument("--prompt", default=None, help="Path to custom system prompt file")
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
        return args.term, [], f"Term '{args.term}' not found in provisions.json"
        
    raw_statute = provisions[args.term]["raw"]
    
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
            ndcg_scores.append(ndcg_f([df_st_sorted["label_score"].tolist()], [df_st_sorted["similarity"].tolist()], k=k))

    elif args.type in ["zero_shot", "few_shot", "few_shot_learn", "probabilities"]:
        system_prompt = await retrieve_system_prompt(args.term, args.type, df, raw_statute, args.id_check, provisions, args.prompt)
        if args.verbose == "true":
            print(system_prompt)
        df_gpt = await gpt_label_batch(df, args.type, args.term, args.id_check, raw_statute, system_prompt, verbose=(args.verbose == "true"), model=args.model)
        
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

        y_true = np.array([df_eval["label_score"].values], dtype=float)
        y_score = np.array([df_eval["gpt_label_score"].values], dtype=float)
        
        for i in [10, 20, 40, 100]:
            ndcg_scores.append(ndcg_f(y_true, y_score, k=i))
        df_st_sorted = df_eval.sort_values("gpt_label_score", ascending=False).reset_index(drop=True)

    elif args.type in ["few_shot_learn_full", "zero_shot_full"]:
        system_prompt = await retrieve_system_prompt(args.term, args.type, df, raw_statute, args.id_check, provisions)
        term_ndcg = await evaluate_all_terms_parallel(provisions, system_prompt, args)
        return args.term, [], "OK" # Aggregation handled inside

    if args.verbose == "true" and df_st_sorted is not None:
        topk = 10
        for i, row in df_st_sorted.head(topk).iterrows():
            sim_val = row['similarity'] if 'similarity' in row else row.get('gpt_label_score', 0)
            print(f"[{i+1:>2}] sim={sim_val:.4f}  rel={row['label_score']:<2}  | {row['text']}")

    if ndcg_scores:
        if args.verbose == "true":
            print(f"Method: {args.type} | Term: {args.term}")
            for k, ndcg in zip([10, 20, 40, 100], ndcg_scores):
                print(f"NDCG@{k}: {ndcg}")
    
    if df_st_sorted is not None and args.log == "true":
        label_counts = df_st_sorted["gpt_label_str"].value_counts().to_dict() if "gpt_label_str" in df_st_sorted else {}
        confusion = pd.crosstab(df_st_sorted["label_score"], df_st_sorted["gpt_label_str"]) if "gpt_label_str" in df_st_sorted else None
        log_to_wandb(args.term, args.type, df_st_sorted, ndcg_scores, label_counts, confusion)
            
    return args.term, ndcg_scores, "OK"

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", default="sources/independent_economic_value/independent_economic_value-sentence.json", help="Path to sentences json")
    ap.add_argument("--provisions", default="sources/provisions.json", help="Path to provisions json")
    ap.add_argument("--term", default="independent economic value", help="Provision key to use as query")
    ap.add_argument("--type", default="naive", help ="naive | zero_shot | few_shot | few_shot_learn | probabilities | large_context")
    ap.add_argument("--model", default="gpt-4o-mini", help="Model to use (default: gpt-4o-mini)")
    ap.add_argument("--log", default="false", help="Log to wandb")
    ap.add_argument("--verbose", default="false", help="Verbose output")
    ap.add_argument("--id_check", default="false", help="Run ID check")
    ap.add_argument("--prompt", default=None, help="Path to custom system prompt file")
    args = ap.parse_args()

    asyncio.run(run_ranking_engine(args))


if __name__ == "__main__":
    main()