import subprocess
import re
import sys
import argparse
from pathlib import Path

HARD_CASES = [
    "nonindustrial use",
    "hazardous liquid",
    "small manufacturer",
    "identifying particular",
    "preexisting work",
    "hybrid instrument",
    "dischargeable consumer debt",
    "essential step",
    "familiar symbol",
    "common business purpose"
]

def run_evaluation(prompt_path, workers, terms=None):
    print(f"\n>>> Running evaluation for: {prompt_path}")
    cmd = [
        sys.executable, "run_multi.py",
        "--type", "probabilities",
        "--prompt", prompt_path,
        "--workers", str(workers),
        "--runs", "1"
    ]
    
    if terms:
        cmd.extend(["--terms", ",".join(terms)])
    
    # Use Popen to stream output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    output_lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)
    
    process.wait()
    
    if process.returncode != 0:
        print(f"\nError running evaluation (Exit {process.returncode})")
        return None
    
    # Extract the results table from captured output
    results = {}
    lines = [l.strip() for l in output_lines]
    
    # Find the table start
    table_start = -1
    for i, line in enumerate(lines):
        if "TERM" in line and "NDCG@10" in line:
            table_start = i + 2
            break
            
    if table_start == -1:
        print("Could not find results table in output.")
        print("--- FULL OUTPUT ---")
        print("".join(output_lines))
        return None
        
    found_any = False
    for line in lines[table_start:]:
        if line.startswith("-") or line.startswith("="):
            break
        if "|" not in line:
            continue
            
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
            
        term = parts[0]
        try:
            ndcg10 = float(parts[1])
            ndcg20 = float(parts[2])
            ndcg100 = float(parts[3])
            results[term] = {
                "ndcg@10": ndcg10,
                "ndcg@20": ndcg20,
                "ndcg@100": ndcg100
            }
            found_any = True
        except ValueError:
            # Check if there's an error message in the status column
            status = parts[4] if len(parts) > 4 else "Unknown error"
            print(f"  ! Error for {term}: {status}")
            continue
            
    if not found_any:
        print("Table found but no successful results were parsed.")
        return None
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark two zero-shot prompts.")
    parser.add_argument("--new", default="system_prompts/zero_shot.txt", help="Path to the new prompt")
    parser.add_argument("--old", default="system_prompts/zero_shot_previous_best.txt", help="Path to the old prompt")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent workers")
    parser.add_argument("--hard", action="store_true", help="Only test against the hardest cases")
    parser.add_argument("--terms", default=None, help="Comma-separated list of terms to test")
    
    args = parser.parse_args()
    
    selected_terms = None
    if args.hard:
        selected_terms = HARD_CASES
    elif args.terms:
        selected_terms = [t.strip() for t in args.terms.split(",")]

    new_results = run_evaluation(args.new, args.workers, selected_terms)
    old_results = run_evaluation(args.old, args.workers, selected_terms)
    
    if not new_results or not old_results:
        print("Failed to gather results for one or both prompts.")
        return

    # Compare results
    terms = sorted(list(set(new_results.keys()) & set(old_results.keys())))
    
    print("\n" + "="*90)
    print(f"{'TERM':<40} | {'OLD (N10)':<10} | {'NEW (N10)':<10} | {'DELTA':<10}")
    print("-" * 90)
    
    avg_old = 0
    avg_new = 0
    count = 0
    
    for term in terms:
        old_val = old_results[term]["ndcg@10"]
        new_val = new_results[term]["ndcg@10"]
        delta = new_val - old_val
        
        print(f"{term:<40} | {old_val:.4f}     | {new_val:.4f}     | {delta:+.4f}")
        
        avg_old += old_val
        avg_new += new_val
        count += 1
        
    print("-" * 90)
    if count > 0:
        avg_old /= count
        avg_new /= count
        print(f"{'AVERAGE':<40} | {avg_old:.4f}     | {avg_new:.4f}     | {avg_new - avg_old:+.4f}")
    print("="*90)

if __name__ == "__main__":
    main()
