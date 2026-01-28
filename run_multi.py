import subprocess
import re
import json
import argparse
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def parse_rank_output(stdout: str) -> Tuple[Optional[str], Dict[int, float]]:
    """Parses stdout from rank.py to extract Term name and NDCG scores."""
    term = None
    ndcg = {}

    for line in stdout.splitlines():
        line = line.strip()

    for line in stdout.splitlines():
        line = line.strip()

        # Format 1: "Method: ... | Term: ..."
        if line.startswith("Method:"):
            parts = line.split("|")
            for part in parts:
                part = part.strip()
                if part.startswith("Term:"):
                    term = part.replace("Term:", "").strip()

        # Format 2: "✓ Completed large_context for term: ... | NDCG@10: ..."
        elif line.startswith("✓ Completed large_context for term:"):
            # e.g. "✓ Completed large_context for term: accommodation trade | NDCG@10: 0.6039"
            parts = line.split("|")
            # Parse term
            first_part = parts[0].strip()
            if "term:" in first_part:
                term = first_part.split("term:")[1].strip()
                
            # Parse NDCG from subsequent parts
            for part in parts[1:]:
                score_part = part.strip()
                if score_part.startswith("NDCG@"):
                    m = re.match(r"NDCG@(\d+):\s*([0-9.]+)", score_part)
                    if m:
                        k = int(m.group(1))
                        val = float(m.group(2))
                        ndcg[k] = val

        # Parse standalone NDCG lines "NDCG@10: 0.1234" (Method 1 or 2 continuation)
        elif line.startswith("NDCG@"):
             # e.g. "NDCG@10: 0.8247..."
             m = re.match(r"NDCG@(\d+):\s*([0-9.]+)", line)
             if m:
                 k = int(m.group(1))
                 val = float(m.group(2))
                 ndcg[k] = val

    return term, ndcg

def run_rank_for_term(term: str, args) -> dict:
    """
    Runs rank.py for a specific term and returns results dict.
    Constructs paths dynamically: sources/{term_snake}/{term_snake}-sentence.json
    """
    snake_term = term.replace(" ", "_")
    base_dir = Path("sources")
    
    # Try predictable path first
    sent_path = base_dir / snake_term / f"{snake_term}-sentence.json"
    
    # If not found, try to find any *-sentence.json in that folder
    if not sent_path.exists():
        folder = base_dir / snake_term
        if folder.exists() and folder.is_dir():
            candidates = list(folder.glob("*-sentence.json"))
            if candidates:
                sent_path = candidates[0]
            else:
                return {"term": term, "error": "No sentence file found"}
        else:
             return {"term": term, "error": f"Folder not found: {folder}"}

    # Construct command
    cmd = [
        sys.executable, "rank.py",
        "--term", term,
        "--type", args.type,
        "--sentences", str(sent_path),
        "--provisions", args.provisions,
        "--verbose", args.verbose
    ]
    
    # Run multiple times if requested
    run_results = []
    
    for i in range(args.runs):
        try:
            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            # Save output to log file
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"{snake_term}_run_{i}.log"
            
            with log_file.open("w", encoding="utf-8") as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write("-" * 40 + "\n")
                f.write(completed.stdout)
                if completed.stderr:
                    f.write("\n" + "-" * 40 + "\nSTDERR:\n")
                    f.write(completed.stderr)

            parsed_term, ndcg = parse_rank_output(completed.stdout)
            
            if not ndcg:
                # Fallback: maybe rank.py failed silently or didn't output NDCG
                tqdm.write(f"[WARN] No NDCG found for {term}. See log: {log_file}")
                run_results.append({})
            else:
                run_results.append(ndcg)
                
        except subprocess.CalledProcessError as e:
            # Log error output too
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"{snake_term}_run_{i}_ERROR.log"
            with log_file.open("w", encoding="utf-8") as f:
                 f.write(f"Command: {' '.join(cmd)}\n")
                 f.write(f"Error Code: {e.returncode}\n")
                 f.write(e.stdout or "")
                 f.write(e.stderr or "")

            return {
                "term": term,
                "error": f"Process failed (exit {e.returncode}). Log: {log_file}",
                "stderr": e.stderr
            }
        except Exception as e:
            return {"term": term, "error": str(e)}

    # Aggregate results
    if not run_results:
        return {"term": term, "error": "No valid results captured"}
        
    # Calculate average NDCG
    avg_scores = {}
    valid_runs = [r for r in run_results if r]
    
    if valid_runs:
        for k in [10, 20, 40, 100]:
            vals = [r.get(k, 0.0) for r in valid_runs]
            avg_scores[k] = sum(vals) / len(vals)
            
    return {
        "term": term,
        "runs": len(valid_runs),
        "scores": avg_scores,
        "detail": run_results
    }

def main():
    parser = argparse.ArgumentParser(description="Run rank.py evaluation across all terms.")
    parser.add_argument("--type", default="large_context", help="Ranking method type")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per term")
    parser.add_argument("--provisions", default="sources/provisions.json", help="Path to provisions.json")
    parser.add_argument("--verbose", default="false", help="Pass verbose flag to rank.py")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers")
    
    args = parser.parse_args()
    
    # Load terms
    prov_path = Path(args.provisions)
    if not prov_path.exists():
        print(f"Error: {prov_path} not found.")
        sys.exit(1)
        
    with prov_path.open("r", encoding="utf-8") as f:
        provisions = json.load(f)
    
    terms = list(provisions.keys())
    print(f"Found {len(terms)} terms in provisions.json")
    print(f"Mode: {args.type} | Runs: {args.runs} | Workers: {args.workers}")
    print("-" * 60)
    
    results = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(run_rank_for_term, term, args): term for term in terms}
        
        for future in tqdm(as_completed(future_map), total=len(terms), desc="Evaluating"):
            res = future.result()
            results.append(res)
            
            # Identify term for logging
            t_name = res.get("term", "Unknown")
            if "error" in res:
                # tqdm.write(f"!! {t_name}: {res['error']}")
                pass
            else:
                s = res.get("scores", {})
                n10 = s.get(10, 0.0)
                n100 = s.get(100, 0.0)
                tqdm.write(f"✓ {t_name:<30} | NDCG@10: {n10:.4f} | NDCG@100: {n100:.4f}")

    # Final summary
    print("\n" + "="*110)
    print(f"{'TERM':<40} | {'NDCG@10':<10} | {'NDCG@20':<10} | {'NDCG@100':<10} | {'Status'}")
    print("-" * 110)
    
    # Sort by term name
    results.sort(key=lambda x: x.get("term", ""))
    
    stats_ndcg10 = []
    stats_ndcg100 = []
    
    for r in results:
        name = r.get("term", "Unknown")
        if "error" in r:
            print(f"{name:<40} | {'-':<10} | {'-':<10} | {'-':<10} | ERR: {r['error']}")
        else:
            s = r.get("scores", {})
            n10 = s.get(10, 0.0)
            n20 = s.get(20, 0.0)
            n100 = s.get(100, 0.0)
            stats_ndcg10.append(n10)
            stats_ndcg100.append(n100)
            print(f"{name:<40} | {n10:.4f}      | {n20:.4f}      | {n100:.4f}      | OK ({r['runs']} runs)")
            
    print("-" * 110)
    if stats_ndcg10:
        avg_10 = sum(stats_ndcg10) / len(stats_ndcg10)
        avg_100 = sum(stats_ndcg100) / len(stats_ndcg100)
        print(f"AVERAGE NDCG@10  across {len(stats_ndcg10)} successful terms: {avg_10:.4f}")
        print(f"AVERAGE NDCG@100 across {len(stats_ndcg100)} successful terms: {avg_100:.4f}")
    print("="*110)

if __name__ == "__main__":
    main()