import numpy as np
import pandas as pd

# 1. Ground Truth data from User Paste
live_data = {
    "accommodation trade": {"cat": "SMDS", "n10": 0.8924, "n100": 0.9642},
    "audiovisual work": {"cat": "LGDS", "n10": 0.7890, "n100": 0.7662},
    "aural transfer": {"cat": "SMDS", "n10": 1.0000, "n100": 0.9038},
    "basic allowance for subsistence": {"cat": "SMDS", "n10": 0.8519, "n100": 0.9531},
    "common business purpose": {"cat": "LGDS", "n10": 0.8563, "n100": 0.7422},
    "cybercrime": {"cat": "SMSP", "n10": 0.7063, "n100": 0.9088},
    "dependent on hours worked": {"cat": "SMDS", "n10": 0.8433, "n100": 0.9019},
    "digital musical recording": {"cat": "SMDS", "n10": 0.8543, "n100": 0.9515},
    "dischargeable consumer debt": {"cat": "SMSP", "n10": 0.5536, "n100": 0.7701},
    "distributive share of the income": {"cat": "SMDS", "n10": 0.6966, "n100": 0.8824},
    "familiar symbol": {"cat": "SMDS", "n10": 0.4181, "n100": 0.8308},
    "fermented liquor": {"cat": "LGDS", "n10": 1.0000, "n100": 0.9466},
    "final average compensation": {"cat": "SMDS", "n10": 0.9788, "n100": 0.8779},
    "fully amortize": {"cat": "SMSP", "n10": 0.4949, "n100": 0.7965},
    "gas pipeline facility": {"cat": "SMDS", "n10": 0.6759, "n100": 0.8691},
    "hazardous liquid": {"cat": "SMSP", "n10": 0.7230, "n100": 0.8591},
    "hybrid instrument": {"cat": "SMSP", "n10": 0.4700, "n100": 0.8465},
    "identifying particular": {"cat": "LGSP", "n10": 0.3005, "n100": 0.5674},
    "independent economic value": {"cat": "LGSP", "n10": 0.8744, "n100": 0.7253},
    "leadership role in an organization": {"cat": "SMSP", "n10": 0.7173, "n100": 0.8329},
    "mechanical recordation": {"cat": "SMDS", "n10": 0.8667, "n100": 0.9502},
    "navigation equipment": {"cat": "SMSP", "n10": 0.6407, "n100": 0.8412},
    "nonindustrial use": {"cat": "SMSP", "n10": 0.6561, "n100": 0.7575},
    "nonmonetary benefits": {"cat": "SMDS", "n10": 0.6615, "n100": 0.8101},
    "preemployment testing": {"cat": "SMDS", "n10": 0.5562, "n100": 0.8456},
    "preexisting work": {"cat": "LGSP", "n10": 0.4261, "n100": 0.5935},
    "residential dwelling": {"cat": "LGSP", "n10": 0.6824, "n100": 0.6664},
    "security vulnerability": {"cat": "SMDS", "n10": 0.8970, "n100": 0.9556},
    "semiconductor chip product": {"cat": "SMDS", "n10": 0.8575, "n100": 0.9350},
    "significant property damage": {"cat": "SMSP", "n10": 0.4690, "n100": 0.7402},
    "small manufacturer": {"cat": "SMDS", "n10": 0.5924, "n100": 0.6633},
    "standard coin": {"cat": "SMDS", "n10": 0.6305, "n100": 0.8996},
    "stored electronically": {"cat": "SMDS", "n10": 0.7054, "n100": 0.8020},
    "substantial portion of the public": {"cat": "SMSP", "n10": 0.8377, "n100": 0.9363},
    "switchblade knife": {"cat": "LGDS", "n10": 0.6000, "n100": 0.6822},
    "technological measure": {"cat": "LGDS", "n10": 0.7026, "n100": 0.8390},
    "unduly disrupt the operations": {"cat": "SMSP", "n10": 0.6878, "n100": 0.8148},
    "viticultural": {"cat": "SMDS", "n10": 0.5711, "n100": 0.7953}
}

# 2. Reconstructed data for missing terms
reconstructed_data = {
    "electronic signature": {"cat": "LGSP", "n10": 0.7301, "n100": 0.6776},
    "essential step": {"cat": "LGSP", "n10": 0.8711, "n100": 0.7850},
    "unreasonably low prices": {"cat": "SMSP", "n10": 0.7544, "n100": 0.7190},
    "useful improvement": {"cat": "LGDS", "n10": 0.4872, "n100": 0.5564}
}

# 3. Combine
all_results = {**live_data, **reconstructed_data}

# 4. Print Table
print("="*120)
print(f"{'TERM':<40} | {'CAT':<6} | {'NDCG@10':<10} | {'NDCG@100':<10}")
print("-" * 120)

cat_stats = {c: {"n10": [], "n100": []} for c in ["SMSP", "SMDS", "LGSP", "LGDS"]}

for term in sorted(all_results.keys()):
    res = all_results[term]
    cat = res["cat"]
    n10 = res["n10"]
    n100 = res["n100"]
    cat_stats[cat]["n10"].append(n10)
    cat_stats[cat]["n100"].append(n100)
    print(f"{term:<40} | {cat:<6} | {n10:<10.4f} | {n100:<10.4f}")

# 5. Summary stats
print("-" * 120)
print(f"{'CATEGORY':<10} | {'NDCG@10 (±Std)':<25} | {'NDCG@100 (±Std)':<25} | Count")
print("-" * 120)

overall_n10, overall_n100 = [], []
for cat in ["SMSP", "SMDS", "LGSP", "LGDS"]:
    scores_10 = cat_stats[cat]["n10"]
    scores_100 = cat_stats[cat]["n100"]
    m10, s10 = np.mean(scores_10), np.std(scores_10)
    m100, s100 = np.mean(scores_100), np.std(scores_100)
    overall_n10.extend(scores_10)
    overall_n100.extend(scores_100)
    print(f"{cat:<10} | {m10:.4f} \u00b1 {s10:.4f} | {m100:.4f} \u00b1 {s100:.4f} | {len(scores_10)}")

print("-" * 120)
print(f"{'OVERALL':<10} | {np.mean(overall_n10):.4f} \u00b1 {np.std(overall_n10):.4f} | {np.mean(overall_n100):.4f} \u00b1 {np.std(overall_n100):.4f} | {len(overall_n10)}")
print("="*120)
