import json
import argparse
import sys
import os
import asyncio
import contextlib
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm.asyncio import tqdm
import rank

async def run_rank_for_term_async(term: str, args) -> dict:
    snake_term = term.replace(" ", "_")
    base_dir = Path("sources")
    
    sent_path = base_dir / snake_term / f"{snake_term}-sentence.json"
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

    # Prepare arguments for rank.run_ranking_engine
    class RankArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    rank_args = RankArgs(
        sentences=str(sent_path),
        provisions=args.provisions,
        term=term,
        type=args.type,
        model=args.model if hasattr(args, 'model') else "gpt-4o-mini",
        log=args.log,
        verbose=args.verbose,
        id_check=args.id_check if hasattr(args, 'id_check') else "false",
        prompt=args.prompt
    )

    run_results = []
    for i in range(args.runs):
        log_file = None
        if args.log == "true":
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file_path = log_dir / f"{term.replace(' ', '_')}_run_{i}.log"
            log_file = open(log_file_path, "w", encoding="utf-8")
            
        try:
            # Call the ranking engine directly
            if log_file:
                with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                    # We also need to print the command for reference
                    print(f"Term: {term} | Model: {args.model} | Mode: {args.type}")
                    term_name, scores, status = await rank.run_ranking_engine(rank_args)
            else:
                term_name, scores, status = await rank.run_ranking_engine(rank_args)
            
            if log_file:
                log_file.close()
                
            if status != "OK":
                return {"term": term, "error": status}
            
            # Map [10, 20, 40, 100] to dict
            ndcg_dict = {k: v for k, v in zip([10, 20, 40, 100], scores)}
            run_results.append(ndcg_dict)
                
        except Exception as e:
            return {"term": term, "error": str(e)}

    if not run_results:
        return {"term": term, "error": "No valid results captured"}
        
    avg_scores = {}
    for k in [10, 20, 40, 100]:
        vals = [r.get(k, 0.0) for r in run_results]
        avg_scores[k] = sum(vals) / len(vals)
            
    return {
        "term": term,
        "runs": len(run_results),
        "scores": avg_scores,
        "detail": run_results
    }

async def async_main():
    parser = argparse.ArgumentParser(description="Run rank.py evaluation in-process (Fast).")
    parser.add_argument("--type", default="zero_shot", help="Ranking method type")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per term")
    parser.add_argument("--provisions", default="sources/provisions.json", help="Path to provisions.json")
    parser.add_argument("--verbose", default="false", help="Pass verbose flag to rank.py")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent workers (async limit)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use (e.g., gpt-4o, gpt-4.5-preview)")
    parser.add_argument("--prompt", default=None, help="Path to custom system prompt file")
    parser.add_argument("--terms", default=None, help="Comma-separated list of specific terms to run")
    parser.add_argument("--log", default="false", help="Enable logging to files in logs/ directory")

    
    args = parser.parse_args()
    
    prov_path = Path(args.provisions)
    if not prov_path.exists():
        print(f"Error: {prov_path} not found.")
        sys.exit(1)
        
    with prov_path.open("r", encoding="utf-8") as f:
        provisions = json.load(f)
    
    terms = list(provisions.keys())
    if args.terms:
        selected_terms = [t.strip() for t in args.terms.split(",")]
        terms = [t for t in terms if t in selected_terms]
        print(f"Filtering to {len(terms)} specified terms.")
    
    print(f"Found {len(terms)} terms in provisions.json")
    print(f"Mode: {args.type} | Runs: {args.runs} | Async Limit: {args.workers}")
    print("-" * 60)
    
    # Control concurrency using a Semaphore
    sem = asyncio.Semaphore(args.workers)
    
    async def sem_task(term):
        async with sem:
            return await run_rank_for_term_async(term, args)

    tasks = [sem_task(term) for term in terms]
    results = []
    pbar = tqdm(total=len(terms), desc="Benchmarking")
    
    for f in asyncio.as_completed(tasks):
        res = await f
        results.append(res)
        
        t_name = res.get("term", "Unknown")
        pbar.set_postfix(term=t_name[:15])
        pbar.update(1)
        
        if "error" in res:
            tqdm.write(f"✗ {t_name:<30} | ERROR: {res['error']}")

    # Category Mappings
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
    
    # Reverse mapping for lookups
    term_to_cat = {}
    for cat, items in CATEGORIES.items():
        for t in items:
            term_to_cat[t] = cat

    # Final summary
    print("\n" + "="*130)
    print(f"{'TERM':<40} | {'CAT':<6} | {'NDCG@10':<10} | {'NDCG@20':<10} | {'NDCG@100':<10} | {'Status'}")
    print("-" * 130)
    
    results.sort(key=lambda x: x.get("term", ""))
    
    import numpy as np
    cat_stats = {cat: {"n10": [], "n100": []} for cat in CATEGORIES.keys()}
    overall_n10 = []
    overall_n100 = []
    
    for r in results:
        name = r.get("term", "Unknown")
        cat = term_to_cat.get(name, "OTHER")
        if "error" in r:
            print(f"{name:<40} | {cat:<6} | {'-':<10} | {'-':<10} | {'-':<10} | ERR: {r['error']}")
        else:
            s = r.get("scores", {})
            n10 = s.get(10, 0.0)
            n20 = s.get(20, 0.0)
            n100 = s.get(100, 0.0)
            
            if cat in cat_stats:
                cat_stats[cat]["n10"].append(n10)
                cat_stats[cat]["n100"].append(n100)
            overall_n10.append(n10)
            overall_n100.append(n100)
            
            print(f"{name:<40} | {cat:<6} | {n10:.4f}      | {n20:.4f}      | {n100:.4f}      | OK ({r['runs']} runs)")
            
    print("-" * 130)
    print(f"{'CATEGORY':<10} | {'NDCG@10 (±Std)':<25} | {'NDCG@100 (±Std)':<25} | Count")
    print("-" * 130)
    
    for cat in ["SMSP", "SMDS", "LGSP", "LGDS"]:
        scores_10 = cat_stats[cat]["n10"]
        scores_100 = cat_stats[cat]["n100"]
        if scores_10:
            m10, s10 = np.mean(scores_10), np.std(scores_10)
            m100, s100 = np.mean(scores_100), np.std(scores_100)
            print(f"{cat:<10} | {m10:.4f} ± {s10:.4f} | {m100:.4f} ± {s100:.4f} | {len(scores_10)}")
        else:
            print(f"{cat:<10} | {'-':<25} | {'-':<25} | 0")
            
    if overall_n10:
        o_m10, o_s10 = np.mean(overall_n10), np.std(overall_n10)
        o_m100, o_s100 = np.mean(overall_n100), np.std(overall_n100)
        print("-" * 130)
        print(f"{'OVERALL':<10} | {o_m10:.4f} ± {o_s10:.4f} | {o_m100:.4f} ± {o_s100:.4f} | {len(overall_n10)}")
    
    print("="*130)

if __name__ == "__main__":
    asyncio.run(async_main())