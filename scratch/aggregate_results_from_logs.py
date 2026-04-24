import os
import re
import numpy as np

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

term_to_cat = {}
for cat, terms in CATEGORIES.items():
    for t in terms:
        term_to_cat[t] = cat

log_dir = "logs"
results = {}

for filename in os.listdir(log_dir):
    if filename.endswith("_run_0.log"):
        # term name from filename: "accommodation_trade_run_0.log" -> "accommodation trade"
        term_slug = filename.replace("_run_0.log", "")
        term_name = term_slug.replace("_", " ")
        
        path = os.path.join(log_dir, filename)
        with open(path, "r") as f:
            content = f.read()
            n10_match = re.search(r"NDCG@10:\s*([\d.]+)", content)
            n100_match = re.search(r"NDCG@100:\s*([\d.]+)", content)
            
            if n10_match and n100_match:
                results[term_name] = {
                    "n10": float(n10_match.group(1)),
                    "n100": float(n100_match.group(1))
                }

print("\n" + "="*110)
print(f"{'TERM':<40} | {'CAT':<6} | {'NDCG@10':<10} | {'NDCG@100':<10}")
print("-" * 110)

cat_stats = {cat: {"n10": [], "n100": []} for cat in CATEGORIES.keys()}
overall_n10 = []
overall_n100 = []

for term in sorted(results.keys()):
    cat = term_to_cat.get(term, "OTHER")
    vals = results[term]
    n10, n100 = vals["n10"], vals["n100"]
    print(f"{term:<40} | {cat:<6} | {n10:<10.4f} | {n100:<10.4f}")
    
    if cat in cat_stats:
        cat_stats[cat]["n10"].append(n10)
        cat_stats[cat]["n100"].append(n100)
    overall_n10.append(n10)
    overall_n100.append(n100)

print("-" * 110)
print(f"{'CATEGORY':<10} | {'NDCG@10 (±Std)':<25} | {'NDCG@100 (±Std)':<25} | Count")
print("-" * 110)

for cat in ["SMSP", "SMDS", "LGSP", "LGDS"]:
    scores_10 = cat_stats[cat]["n10"]
    scores_100 = cat_stats[cat]["n100"]
    if scores_10:
        m10, s10 = np.mean(scores_10), np.std(scores_10)
        m100, s100 = np.mean(scores_100), np.std(scores_100)
        print(f"{cat:<10} | {m10:.4f} \u00b1 {s10:.4f} | {m100:.4f} \u00b1 {s100:.4f} | {len(scores_10)}")
    else:
        print(f"{cat:<10} | {'-':<25} | {'-':<25} | 0")

if overall_n10:
    o_m10, o_s10 = np.mean(overall_n10), np.std(overall_n10)
    o_m100, o_s100 = np.mean(overall_n100), np.std(overall_n100)
    print("-" * 110)
    print(f"{'OVERALL':<10} | {o_m10:.4f} \u00b1 {o_s10:.4f} | {o_m100:.4f} \u00b1 {o_s100:.4f} | {len(overall_n10)}")

print("="*110 + "\n")
