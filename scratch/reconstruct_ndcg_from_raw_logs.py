import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import ndcg_score

CATEGORIES = {
    "SMSP": ["cybercrime", "dischargeable consumer debt", "fully amortize", "hazardous liquid", "hybrid instrument", 
             "leadership role in an organization", "navigation equipment", "nonindustrial use", 
             "significant property damage", "substantial portion of the public", "unduly disrupt the operations", 
             "unreasonably low prices"],
    "LGSP": ["electronic signature", "essential step", "identifying particular", "independent economic value", 
             "residential dwelling", "preexisting work"],
    "SMDS": ["accommodation trade", "aural transfer", "basic allowance for subsistence", "dependent on hours worked", 
             "digital musical recording", "distributive share of the income", "familiar symbol", 
             "final average compensation", "gas pipeline facility", "mechanical recordation", "nonmonetary benefits", 
             "preemployment testing", "security vulnerability", "small manufacturer", "semiconductor chip product", 
             "standard coin", "stored electronically", "viticultural"],
    "LGDS": ["audiovisual work", "common business purpose", "fermented liquor", "switchblade knife", 
             "technological measure", "useful improvement"]
}

def repair_json(text):
    # Numerical hallucination repair
    num_map = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
    }
    for word, digit in num_map.items():
        text = re.sub(rf"0\.\s*{word}", f"0.{digit}", text, flags=re.IGNORECASE)
        text = re.sub(rf":\s*{word}", f": {digit}", text, flags=re.IGNORECASE)
    
    # Handle leading zeros (e.g., 00 -> 0, or 09 -> 0.9 if it looks like a probability)
    # If it's 0 followed by digits, and it's a probability (0-1), 09 likely means 0.9
    text = re.sub(r':\s*0([1-9])(?!\.)', r': 0.\1', text)
    # Double zeros like 00 should just be 0
    text = re.sub(r':\s*00(?!\.)', r': 0', text)
    
    # Handle spacing in decimals (0. 9 -> 0.9)
    text = re.sub(r"0\.\s*(\d)", r"0.\1", text)
    
    # Handle trailing letters in numbers (e.g. 0.9ty -> 0.9, 0.00abc -> 0.00)
    text = re.sub(r':\s*(\d+\.?\d*)[a-zA-Z]+', r': \1', text)
    
    return text

def extract_json_from_log(log_path):
    all_data = []
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find all "Original Text snippet" blocks
    # We look for something that starts with { and likely ends with } ...
    # This is a bit tricky with regex, so we'll look for the markers
    snippets = re.findall(r"Original Text snippet \(first 2000 chars\): (\{.*?)\.\.\.", content, re.DOTALL)
    if not snippets:
        # Fallback for old logs without the "first 2000 chars" marker
        snippets = re.findall(r"Original Text snippet: (\{.*?)\.\.\.", content, re.DOTALL)

    for snip in snippets:
        try:
            repaired = repair_json(snip)
            # Find the last } to close the object if it was truncated
            last_brace = repaired.rfind('}')
            if last_brace != -1:
                repaired = repaired[:last_brace+1]
            
            data = json.loads(repaired)
            if isinstance(data, dict) and "results" in data:
                all_data.extend(data["results"])
            elif isinstance(data, list):
                all_data.extend(data)
        except Exception as e:
            # print(f"  Warning: Could not parse snippet in {log_path}: {e}")
            pass
    return all_data

def get_labels(term):
    # Try to find the sentence file
    term_slug = term.replace(" ", "_")
    paths = [
        Path(f"sources/{term_slug}/{term_slug}-sentence.json"),
        Path(f"sources/{term}/{term}-sentence.json"),
    ]
    for p in paths:
        if p.exists():
            with open(p, "r") as f:
                data = json.load(f)
                return {sid: entry.get("label_score", 0) for sid, entry in data.items()}
    return {}

def calculate_reconstructed_ndcg(term, gpt_data, gold_labels):
    if not gpt_data or not gold_labels:
        return None
    
    rows = []
    for item in gpt_data:
        sid = item.get("sentence_id", item.get("id"))
        if not sid or sid not in gold_labels:
            continue
        
        probs = item.get("probs", item.get("label"))
        if isinstance(probs, str):
            # Fallback for label strings
            val = probs.lower()
            probs = {"no value": 0, "potential value": 0, "certain value": 0, "high value": 0}
            probs[val] = 1.0

        if not isinstance(probs, dict): continue

        def safe_f(v): 
            try: return float(v)
            except: return 0.0

        ev = (safe_f(probs.get("no value", 0)) * 0 +
              safe_f(probs.get("potential value", 0)) * 1 +
              safe_f(probs.get("certain value", 0)) * 2 +
              safe_f(probs.get("high value", 0)) * 3)
        
        rows.append({
            "sentence_id": sid,
            "gold": gold_labels[sid],
            "score": ev
        })
    
    if not rows: return None
    df = pd.DataFrame(rows)
    # Average duplicate IDs if any
    df = df.groupby("sentence_id").mean().reset_index()
    
    y_true = [df["gold"].tolist()]
    y_score = [df["score"].tolist()]
    
    res = {}
    for k in [10, 20, 100]:
        try:
            res[k] = ndcg_score(y_true, y_score, k=k)
        except:
            res[k] = 0.0
    return res

def main():
    log_dir = Path("logs")
    term_to_cat = {t: c for c, terms in CATEGORIES.items() for t in terms}
    
    final_results = {}
    
    print("Reconstructing results from logs...")
    for log_path in log_dir.glob("*_run_0.log"):
        term_slug = log_path.stem.replace("_run_0", "")
        term_name = term_slug.replace("_", " ")
        
        # 1. Try to find existing NDCG in the log first
        with open(log_path, "r") as f:
            content = f.read()
            # Standard output format
            match_10 = re.search(r"NDCG@10:\s*([\d.]+)", content, re.IGNORECASE)
            match_100 = re.search(r"NDCG@100:\s*([\d.]+)", content, re.IGNORECASE)
            
            # W&B summary format (e.g. ndcg@10 0.4871)
            if not (match_10 and match_100):
                match_10 = re.search(r"ndcg@10\s+([\d.]+)", content)
                match_100 = re.search(r"ndcg@100\s+([\d.]+)", content)

            if match_10 and match_100:
                final_results[term_name] = {10: float(match_10.group(1)), 100: float(match_100.group(1))}
                continue
        
        # 2. If not found, try to reconstruct from raw snippets
        gpt_data = extract_json_from_log(log_path)
        if gpt_data:
            gold = get_labels(term_name)
            recon = calculate_reconstructed_ndcg(term_name, gpt_data, gold)
            if recon:
                print(f"  [RECONSTRUCTED] {term_name} ({len(gpt_data)} items)")
                final_results[term_name] = recon

    # Print summary
    print("\n" + "="*120)
    print(f"{'TERM':<40} | {'CAT':<6} | {'NDCG@10':<10} | {'NDCG@100':<10} | {'Status'}")
    print("-" * 120)
    
    all_n10, all_n100 = [], []
    cat_stats = {c: {"n10": [], "n100": []} for c in CATEGORIES}

    for term in sorted(term_to_cat.keys()):
        cat = term_to_cat[term]
        if term in final_results:
            n10 = final_results[term][10]
            n100 = final_results[term][100]
            print(f"{term:<40} | {cat:<6} | {n10:<10.4f} | {n100:<10.4f} | OK")
            all_n10.append(n10)
            all_n100.append(n100)
            cat_stats[cat]["n10"].append(n10)
            cat_stats[cat]["n100"].append(n100)
        else:
            print(f"{term:<40} | {cat:<6} | {'-':<10} | {'-':<10} | MISSING")

    print("-" * 120)
    print(f"{'CATEGORY':<10} | {'NDCG@10 (±Std)':<25} | {'NDCG@100 (±Std)':<25} | Count")
    print("-" * 120)
    for cat in ["SMSP", "SMDS", "LGSP", "LGDS"]:
        s10s = cat_stats[cat]["n10"]
        s100s = cat_stats[cat]["n100"]
        if s10s:
            print(f"{cat:<10} | {np.mean(s10s):.4f} \u00b1 {np.std(s10s):.4f} | {np.mean(s100s):.4f} \u00b1 {np.std(s100s):.4f} | {len(s10s)}")
    
    if all_n10:
        print("-" * 120)
        print(f"{'OVERALL':<10} | {np.mean(all_n10):.4f} \u00b1 {np.std(all_n10):.4f} | {np.mean(all_n100):.4f} \u00b1 {np.std(all_n100):.4f} | {len(all_n10)}")
    print("="*120 + "\n")

if __name__ == "__main__":
    main()
