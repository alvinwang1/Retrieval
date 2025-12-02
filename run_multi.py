import subprocess
import re
import json
with open("sources/terms.json", "r") as f:
    TERMS = json.load(f)
def parse_rank_output(stdout: str):
    term = None
    ndcg = {}

    for line in stdout.splitlines():
        line = line.strip()

        # Parse term
        if line.startswith("Method:"):
            # e.g. "Method: zero_shot | Term: accommodation trade"
            # split on '|' then extract after "Term:"
            parts = line.split("|")
            for part in parts:
                part = part.strip()
                if part.startswith("Term:"):
                    term = part.replace("Term:", "").strip()

        # Parse NDCG lines
        elif line.startswith("NDCG@"):
            # e.g. "NDCG@10: 0.8247..."
            # use regex or split
            m = re.match(r"NDCG@(\d+):\s*([0-9.]+)", line)
            if m:
                k = int(m.group(1))          # 10, 20, 40, 100
                val = float(m.group(2))
                ndcg[k] = val

    return term, ndcg


all_results = []

for term in TERMS[:5]:
    print("\n========================")
    print(f" Running term: {term}")
    print("========================\n")

    safe_term = term.replace(" ", "_")
    sentences_path = f"sources/{safe_term}/{safe_term}-sentence.json"

    cmd = [
        "python", "rank.py",
        "--type", "few_shot",
        "--term", term,
        "--sentences", sentences_path,
        "--provisions", "sources/provisions.json",
    ]

    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)

    # Optional: still show the child’s stdout
    print(completed.stdout)

    parsed_term, ndcg = parse_rank_output(completed.stdout)
    # Fallback to CLI term if not found, just in case
    parsed_term = parsed_term or term

    all_results.append({
        "term": parsed_term,
        "ndcg": ndcg,
    })

# After loop, you now have all_results with all NDCG values
print("\n==== SUMMARY ====\n")
for result in all_results:
    term = result["term"]
    ndcg = result["ndcg"]
    print(f"Term: {term}")
    for k in sorted(ndcg.keys()):
        print(f"  NDCG@{k}: {ndcg[k]:.4f}")
