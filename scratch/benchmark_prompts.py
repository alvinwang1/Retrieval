import subprocess
import json
import os

terms = ["hazardous liquid", "useful improvement", "accommodation trade", "basic allowance for subsistence", "essential step"]

# Baseline: Old prompt content + Output Format instructions
# Based on 'system_prompts/zero_shot copy.txt'
with open("system_prompts/zero_shot copy.txt", "r") as f:
    old_content = f.read()

# Remove the old output format section (Line 14-25 approx) and add the new one
# To be safe, we'll just prepend a rule about JSON output.
probs_instruction = """
### OUTPUT FORMAT (FOR PROBABILITIES MODE)
You must return a JSON array of objects. Each object must include a "reasoning" field (Chain-of-Thought) and a "probs" field with probabilities for: "no value", "potential value", "certain value", "high value".
"""

baseline_prompt = old_content + "\n" + probs_instruction
adversarial_prompt = open("system_prompts/zero_shot.txt", "r").read()

prompts = {
    "baseline": baseline_prompt,
    "adversarial": adversarial_prompt
}

results = {}

# Save original zero_shot.txt
with open("system_prompts/zero_shot.txt", "r") as f:
    orig_new = f.read()

try:
    for name, content in prompts.items():
        results[name] = {}
        # Swap the prompt file
        with open("system_prompts/zero_shot.txt", "w") as f:
            f.write(content)
        
        for term in terms:
            print(f"Running {name} prompt for {term}...")
            
            cmd = [
                "/opt/anaconda3/bin/python", "rank.py",
                "--term", term,
                "--type", "probabilities",
                "--sentences", f"sources/{term.replace(' ', '_')}/{term.replace(' ', '_')}-sentence.json",
                "--verbose", "false"
            ]
            
            proc = subprocess.run(cmd, capture_output=True, text=True)
            output = proc.stdout
            
            ndcg10 = 0
            for line in output.splitlines():
                if "NDCG@10:" in line:
                    try:
                        ndcg10 = float(line.split(":")[1].strip())
                    except:
                        pass
            
            results[name][term] = ndcg10

finally:
    # Restore original zero_shot.txt
    with open("system_prompts/zero_shot.txt", "w") as f:
        f.write(orig_new)

print("\nComparison Results (NDCG@10) - Probability Mode Used for Both:")
print(f"{'Term':<35} | {'Baseline':<10} | {'Adversarial':<10} | {'Delta':<10}")
print("-" * 75)
for term in terms:
    base = results["baseline"][term]
    adv = results["adversarial"][term]
    delta = adv - base
    print(f"{term:<35} | {base:<10.4f} | {adv:<10.4f} | {delta:<+10.4f}")

avg_base = sum(results["baseline"].values()) / len(terms)
avg_adv = sum(results["adversarial"].values()) / len(terms)
print("-" * 75)
print(f"{'AVERAGE':<35} | {avg_base:<10.4f} | {avg_adv:<10.4f} | {avg_adv - avg_base:<+10.4f}")
