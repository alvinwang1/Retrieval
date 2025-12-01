import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from openai import OpenAI

import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics import ndcg_score
from tqdm import tqdm


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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

MAX_CHARS = 1200
BATCH_SIZE = 25

def _gpt_label_single_batch(df_batch: pd.DataFrame, type: str) -> pd.DataFrame:
    records = [
        {"sentence_id": str(row["sentence_id"]), "text": str(row["text"])[:MAX_CHARS]}
        for _, row in df_batch.iterrows()
    ]
    if type == "zero_shot":
        system_prompt = """
        You are an expert legal text annotator. Your task is to evaluate each sentence for its usefulness in supporting legal argumentation about the meaning of a statutory or regulatory term (the “phrase of interest”). For each sentence, assign BOTH: (1) a categorical label in {"no value", "potential value", "certain value", "high value"} and (2) a numeric score in {0, 1, 2, 3} where 0 = no value, 1 = potential value, 2 = certain value, 3 = high value. Be consistent and conservative. If a sentence is clearly irrelevant, assign “no value”/0. Return ONLY valid JSON in the exact format:
        {
        "items": [
            {"sentence_id": "...", "label": "...", "label_score": ...},
            ...
        ]
        }

        Follow these rules:

        A sentence is valuable only if it contributes information relevant to understanding or arguing about the meaning of the phrase of interest. Evaluate each sentence using this workflow:

        STEP 1: Determine whether the sentence provides information beyond what is already stated in the source statutory provision. If No, label NO VALUE (0) and stop. If Yes, continue.

        STEP 2: Determine whether the phrase of interest is used with a different meaning than in the source provision. If Yes, continue to Step 3. If No, skip to Step 4.

        STEP 3: If the meaning differs, determine whether the meaning in the sentence is still related to the meaning in the provision. If unrelated, label NO VALUE (0). If related, assign POTENTIAL VALUE (1) or CERTAIN VALUE (2) depending on strength of the relation. Stop.

        STEP 4: Determine whether the sentence explicitly elaborates the meaning of the phrase of interest. Explicit elaboration includes: definition; explanation; positive example; negative example; subsumption to a broader category; contrast against a broader category; assignment of features; exclusion of features. If Yes, go to Step 5. If No, go to Step 7.

        STEP 5: If the sentence is attributed to a person with a personal stake in the outcome (e.g., a party, witness), assign CERTAIN VALUE (2). If attributed to an objective source (judge, court, neutral authority), assign HIGH VALUE (3). Stop.

        STEP 7: Determine whether the sentence provides useful context that allows inferring aspects of the meaning (implicit explanation, implicit example, implicit feature assignment or exclusion). If No, assign POTENTIAL VALUE (1) and stop. If Yes, go to Step 8.

        STEP 8: If the sentence with implicit context is attributed to a person with a personal interest, assign POTENTIAL VALUE (1). If attributed to a neutral source, assign CERTAIN VALUE (2). Stop.

        Additional clarifications:
        - Sentences that merely restate or paraphrase the statutory provision, or provide headings or boilerplate references, do NOT provide additional information and must be labeled NO VALUE.
        - The same phrase may have different meanings across statutes or domains; if understanding one meaning does not help understand the statutory meaning, the uses are unrelated and the sentence has NO VALUE.
        - Explicit elaboration directly defines, explains, clarifies, contrasts, or assigns features to the phrase of interest. Implicit elaboration provides facts from which such features can be inferred.
        - Statements by parties, witnesses, or other interested individuals are less objective and therefore always lower-value than statements from courts or neutral authorities.

        Return only valid JSON following the structure above.
        """
    elif type == "few_shot":
        system_prompt = """You are an expert legal text annotator. Your task is to evaluate each sentence for its usefulness in supporting legal argumentation about the meaning of a statutory or regulatory term (the “phrase of interest”). For each sentence, assign BOTH: (1) a categorical label in {"no value", "potential value", "certain value", "high value"} and (2) a numeric score in {0, 1, 2, 3} where 0 = no value, 1 = potential value, 2 = certain value, 3 = high value. Be consistent and conservative. If a sentence is clearly irrelevant, assign “no value”/0. Return ONLY valid JSON in the exact format:
        {
        "items": [
            {"sentence_id": "...", "label": "...", "label_score": ...},
            ...
        ]
        }

        Statutory and regulatory provisions are difficult to understand because legislators must write rules that apply broadly and abstractly. Lawyers therefore often argue about the meaning of vague or open-textured terms. Past uses of the term from case law, legislative history, or secondary sources may be crucial in supporting or challenging a particular interpretation. Not all sentences mentioning the term are equally useful. Your task is to determine whether each sentence contributes information relevant to understanding or arguing about the meaning of the phrase of interest.

        Here is an illustrative statutory excerpt:
        “Enterprise” means the related activities performed [...] by any person or persons for a common business purpose [...]

        Examples of highly useful sentences for argumentation about the meaning:
        - “The fact of common ownership of the two businesses clearly is not sufficient to establish a common business purpose.”
        - “The profit motive is a common business purpose if shared.”

        Examples of sentences that are less useful:
        - “Because the activities of the two businesses are not related and there is no common business purpose, the question of common control is not determinative.”
        - “The defendants weakly challenge the common business purpose conclusion.”

        Your job is to classify sentences based on how useful they are for argumentation about meaning. Use the decision procedure below.

        STEP 1: Determine whether the sentence provides information beyond what is already stated in the source statutory provision. Sentences that only repeat or paraphrase the provision, or that contain only headings or citations, DO NOT add information. Example provision: “No vehicles are allowed in the park.” Examples that do NOT add information:
        - “The provision states that: 'No vehicles are allowed in the park.'”
        - “A vehicle is forbidden from entering the park.”
        - “Motor Vehicles Inc. v. Jane Doe”
        If the sentence does NOT add information → label NO VALUE (0). If it does, continue.

        STEP 2: Determine whether the phrase of interest is used with a different meaning than in the source provision. Example provision: “No vehicles are allowed in the park.” Example different meaning:
        - “Any autonomous vehicle is subject to the approval of the executive committee.”
        Also, terms can have different meanings in different legal domains; “independent economic value” in copyright vs. trade secret law might differ. If meaning is different → Step 3. If same → Step 4.

        STEP 3: If the meaning differs, determine whether it is still related. Examples:
        Related but slightly different:
        - “Any autonomous vehicle is subject to the approval of the executive committee.”
        Unrelated:
        - “A body is a vehicle for a soul.”
        Rule of thumb: If understanding one meaning helps understand the other, they are related. If related → assign POTENTIAL VALUE (1) or CERTAIN VALUE (2). If unrelated → NO VALUE (0). Stop.

        STEP 4: Determine whether the sentence explicitly elaborates the meaning of the phrase of interest. Explicit elaboration includes:

        1. DEFINITION  
        Examples:  
        - “Any mechanical device used for transportation of people or goods is a vehicle.”  
        - “FOIA request means a written request for agency records that reasonably describes the agency records sought…”

        2. EXPLANATION  
        Examples:  
        - “A vehicle usually has wheels, engine and controls.”  
        - “Likewise, activities are 'related' when they are part of a vertical structure such as the manufacturing, warehousing, and retailing of a product…”

        3. POSITIVE EXAMPLE  
        Examples:  
        - “A car is a vehicle.”  
        - “The Act defines 'record' as any item… containing an identifying number, symbol… such as a photograph.”

        4. NEGATIVE EXAMPLE  
        Examples:  
        - “A stroller is not a vehicle.”  
        - “Duty titles used in lieu of names were not 'identifying particulars'…”

        5. SUBSUMPTION  
        Examples:  
        - “A car is a vehicle.”  
        - “Duty titles may be 'identifying particulars' as used in the Privacy Act.”

        6. CONTRAST  
        Examples:  
        - “Not every vehicle is a man-made object.”  
        - “The duty titles do not qualify as identifying particulars.”

        7. FEATURE ASSIGNMENT  
        Examples:  
        - “Some vehicles are fast.”  
        - “The appellee pointed out that duty titles change over time.”

        8. FEATURE EXCLUSION  
        Examples:  
        - “Some vehicles are not large.”  
        - “Object code does not derive independent economic value from its secrecy.”

        If explicit elaboration exists → Step 5. If not → Step 7.

        STEP 5: Determine whether the sentence is attributed to a person who has a personal interest in the litigation outcome (e.g., a party or witness). Example of interested attribution:
        - “The defendant claimed he did not break the rule since roller skates cannot be considered a vehicle.”
        If attributed to an interested speaker → CERTAIN VALUE (2).  
        If neutral (judge, court, expert) → HIGH VALUE (3).  
        Stop.

        STEP 7: Determine whether the sentence provides useful context from which elaboration could be inferred. Implicit elaboration examples:

        Implicit EXPLANATION:  
        - “The vehicle was stripped of its wheels, the engine, and all the controls.”

        Implicit POSITIVE EXAMPLE:  
        - “All the vehicles including the car were parked there.”

        Implicit NEGATIVE EXAMPLE:  
        - “Whereas all the vehicles had to be parked in front of the building, the stroller was allowed in.”

        Implicit SUBSUMPTION or CONTRAST:  
        - “We recognized the car among all the vehicles.”  
        - “Whereas all the vehicles had to be parked in front of the building, the stroller was allowed in.”

        Implicit FEATURE ASSIGNMENT:  
        - “All the fast vehicles were already gone.”

        Implicit FEATURE EXCLUSION:  
        - “All the vehicles that were not large could enter the road.”

        If no useful context → POTENTIAL VALUE (1). Stop.  
        If useful context → Step 8.

        STEP 8: Determine attribution for implicit context. If attributed to an interested person → POTENTIAL VALUE (1). If neutral → CERTAIN VALUE (2). Stop.

        General attribution rule: Statements by parties, witnesses, or people with personal incentives are always less objective, and therefore yield reduced sentence value compared to statements by judges or courts.

        Return only valid JSON following the exact structure above.
        """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(records)},
        ],
        response_format={"type": "json_object"},
    )

    output_text = response.choices[0].message.content
    gpt_labels = json.loads(output_text)

    # Handle both {"items": [...]} and plain [...] just in case
    if isinstance(gpt_labels, dict) and "items" in gpt_labels:
        items = gpt_labels["items"]
    else:
        items = gpt_labels

    df_gpt = pd.DataFrame(items)
    df_gpt["sentence_id"] = df_gpt["sentence_id"].astype(str)
    df_gpt["gpt_label_str"] = df_gpt["label"]
    df_gpt["gpt_label_score"] = df_gpt["label_score"]
    return df_gpt[["sentence_id", "gpt_label_str", "gpt_label_score"]]

def gpt_label_batch(df: pd.DataFrame, type: str) -> pd.DataFrame:
    batches = []
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="GPT batches", unit="batch"):
        sub = df.iloc[start:start + BATCH_SIZE]
        batch_df = _gpt_label_single_batch(sub, type)
        batches.append(batch_df)
    df_gpt = pd.concat(batches, ignore_index=True)
    return df_gpt

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", default="independent_economic_value/independent_economic_value-sentence.json", help="Path to sentences json")
    ap.add_argument("--provisions", default="provisions.json", help="Path to provisions json")
    ap.add_argument("--term", default="independent economic value", help="Provision key to use as query")
    ap.add_argument("--topk", type=int, default=10, help="Top-K to print and evaluate")
    ap.add_argument("--type", default="naive", help ="naive | zero_shot | few_shot")
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
        df_gpt = gpt_label_batch(df, args.type)
        
        # Merge GPT labels back into the original df (which has text + human labels)
        df_merged = df.merge(df_gpt, on="sentence_id", how="left")

        # Keep only rows with a valid human label (>= 0) and a GPT score
        df_eval = df_merged[df_merged["label_score"] >= 0].copy()
        df_eval = df_eval.dropna(subset=["gpt_label_score"])

        # Ensure numeric types
        df_eval["label_score"] = df_eval["label_score"].astype(float)
        df_eval["gpt_label_score"] = df_eval["gpt_label_score"].astype(float)

        # Sort by GPT's predicted relevance
        df_st_sorted = df_eval.sort_values("gpt_label_score", ascending=False).reset_index(drop=True)

        # Prepare arrays for ndcg_score
        y_true = np.array([df_st_sorted["label_score"].values], dtype=float)
        y_score = np.array([df_st_sorted["gpt_label_score"].values], dtype=float)

        # Make sure k doesn't exceed number of docs
        for k in [10, 20, 40, 100]:
            ndcg_scores.append(ndcg_score(y_true, y_score, k=k))



        print(f"Top {args.topk} sentences")

    topk = min(args.topk, len(df_st_sorted))

    if args.type == "naive":
        for i, row in df_st_sorted.head(topk).iterrows():
            print(f"[{i+1:>2}] sim={row['similarity']:.4f}  rel={row['label_score']:<2}  | {row['text']}")
    elif args.type == "zero_shot":
        for i, row in df_st_sorted.head(topk).iterrows():
            # use GPT’s predicted score as “sim”
            print(f"[{i+1:>2}] sim={row['gpt_label_score']:.4f}  rel={row['label_score']:<2}  | {row['text']}")
    print("\nNDCG scores")
    print(f"Method: {args.type}")
    print(f"NDCG@{args.topk}: {ndcg_scores}")
    

    

if __name__ == "__main__":
    main()