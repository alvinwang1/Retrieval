#!/usr/bin/env python3
"""
Fold-aware ModernBERT training (+ instruction prefixing), 5-fold TRAIN / 1-fold TEST:
  - Choose 1 fold as TEST (held out)
  - Train on remaining 5 folds
  - Evaluate on the held-out TEST fold

Adds:
  - --instruction: a string prepended to every example
  - --sep: separator between instruction and sentence

Run example:
  python bert.py \
    --test_fold 2 --epochs 3 --batch_size 8 --max_length 256 \
    --cpu_only false \
    --instruction "You are an expert legal text evaluator. Label the sentence as no value, potential value, certain value, or high value." \
    --sep "\n\n### Sentence:\n" \
    --verbose true
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    ModernBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# -----------------------------
# Your 6-fold split
# -----------------------------
folds: List[List[str]] = [
    # Fold 1
    [
        "navigation equipment",
        "leadership role in an organization",
        "aural transfer",
        "semiconductor chip product",
        "distributive share of the income",
        "preexisting work",
        "audiovisual work",
    ],
    # Fold 2
    [
        "nonindustrial use",
        "significant property damage",
        "nonmonetary benefits",
        "basic allowance for subsistence",
        "stored electronically",
        "independent economic value",
        "technological measure",
    ],
    # Fold 3
    [
        "unduly disrupt the operations",
        "substantial portion of the public",
        "small manufacturer",
        "accommodation trade",
        "standard coin",
        "residential dwelling",
        "common business purpose",
    ],
    # Fold 4
    [
        "hazardous liquid",
        "fully amortize",
        "security vulnerability",
        "familiar symbol",
        "mechanical recordation",
        "electronic signature",
        "fermented liquor",
    ],
    # Fold 5
    [
        "hybrid instrument",
        "unreasonably low prices",
        "gas pipeline facility",
        "preemployment testing",
        "final average compensation",
        "identifying particular",
        "useful improvement",
    ],
    # Fold 6
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

# -----------------------------
# Label mapping
# -----------------------------
LABEL_MAP = {
    "no value": 0,
    "potential value": 1,
    "certain value": 2,
    "high value": 3,
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

# -----------------------------
# Defaults for instruction prefixing
# -----------------------------
DEFAULT_INSTRUCTION = (
    "You are an expert legal text evaluator. "
    "Classify the sentence according to its interpretive value."
)
DEFAULT_SEP = "\n\n### Sentence:\n"


def build_prompt(sentence: str, instruction: str, sep: str) -> str:
    # Simple: prepend instruction before the sentence.
    return f"### Instruction:\n{instruction}{sep}{sentence}"


# -----------------------------
# Data helpers
# -----------------------------
def build_records(sent_json: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for sid, entry in sent_json.items():
        label_str = (entry.get("label", "") or "").strip().lower()
        rows.append(
            {
                "sentence_id": sid,
                "case_id": entry.get("case_id", ""),
                "opinion_id": entry.get("opinion_id", ""),
                "paragraph_id": entry.get("paragraph_id", ""),
                "position": entry.get("position", -1),
                "text": entry.get("text", ""),
                "label": entry.get("label", ""),
                "label_score": LABEL_MAP.get(label_str, -1),
            }
        )
    return pd.DataFrame(rows)


def load_term_df(term: str) -> pd.DataFrame:
    """Loads sources//-sentence.json into a DataFrame."""
    term_slug = term.replace(" ", "_")
    sent_path = Path(f"/content/drive/MyDrive/src/sources/{term_slug}/{term_slug}-sentence.json")
    if not sent_path.exists():
        raise FileNotFoundError(f"File not found: {sent_path}")
    with sent_path.open("r", encoding="utf-8") as f:
        sent_json = json.load(f)
    df = build_records(sent_json)
    if not df.empty:
        df["term"] = term
    return df


def build_fold_df(fold_idx: int) -> pd.DataFrame:
    """fold_idx is 0-based: 0..5"""
    dfs = []
    for term in folds[fold_idx]:
        df = load_term_df(term)
        if not df.empty:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def build_train_test_dfs(test_fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    test_fold is 1-based in [1,6].
    Returns:
      train_df: all folds except test_fold
      test_df: fold test_fold
    """
    if not (1 <= test_fold <= 6):
        raise ValueError(f"test_fold must be in [1,6], got {test_fold}")
    test_idx = test_fold - 1
    test_df = build_fold_df(test_idx)
    train_dfs = []
    for i in range(6):
        if i == test_idx:
            continue
        df = build_fold_df(i)
        if not df.empty:
            train_dfs.append(df)
    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    return train_df, test_df


# -----------------------------
# Dataset (instruction prefixing happens HERE)
# -----------------------------
class ModernBertDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        instruction: str,
        sep: str,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.sep = sep
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        sentence = str(self.texts[idx])
        label = int(self.labels[idx])

        prompt = build_prompt(sentence=sentence, instruction=self.instruction, sep=self.sep)

        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        enc = {k: torch.tensor(v) for k, v in enc.items()}
        enc["labels"] = torch.tensor(label)
        return enc


# -----------------------------
# Training + prediction
# -----------------------------
def train_modernbert(
    train_df: pd.DataFrame,
    max_length: int,
    num_epochs: int,
    batch_size: int,
    output_dir: str,
    instruction: str,
    sep: str,
    cpu_only: bool = True,
) -> Tuple[Trainer, Any]:
    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_labeled = train_df[train_df["label_score"] >= 0].copy()
    if train_labeled.empty:
        raise ValueError("No labeled rows in TRAIN set (label_score >= 0).")

    train_ds = ModernBertDataset(
        train_labeled["text"].tolist(),
        train_labeled["label_score"].astype(int).tolist(),
        tokenizer,
        instruction=instruction,
        sep=sep,
        max_length=max_length,
    )

    model = ModernBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_MAP),
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
    )

    # Training arguments with NO intermediate checkpointing
    common_args = dict(
        output_dir=output_dir,
        save_strategy="no",  # No intermediate saves
        overwrite_output_dir=True,
        learning_rate=2e-5,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        weight_decay=0.01,
        logging_steps=50,
        no_cuda=cpu_only,
        dataloader_pin_memory=False,
        seed=42,
        report_to=["tensorboard"],
        logging_dir=os.path.join(output_dir, "tb"),
    )

    training_args = TrainingArguments(**common_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )

    print(f"[train] output_dir = {output_dir}")
    print("[train] No intermediate checkpoints will be saved (only final model)")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Train without checkpointing
    trainer.train()

    # Save ONLY the final model
    save_dir = os.path.join(output_dir, "final-model")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[train] Saving final model to {save_dir}")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    return trainer, tokenizer


def predict_on_df(
    trainer: Trainer,
    tokenizer,
    df: pd.DataFrame,
    max_length: int,
    instruction: str,
    sep: str,
) -> pd.DataFrame:
    df_lab = df[df["label_score"] >= 0].copy()
    if df_lab.empty:
        raise ValueError("No labeled rows in DF for prediction/eval.")

    ds = ModernBertDataset(
        df_lab["text"].tolist(),
        df_lab["label_score"].astype(int).tolist(),
        tokenizer,
        instruction=instruction,
        sep=sep,
        max_length=max_length,
    )

    preds = trainer.predict(ds)
    logits = preds.predictions
    pred_ids = np.argmax(logits, axis=-1)

    out = df_lab.reset_index(drop=True)
    out["pred_label_score"] = pred_ids
    out["pred_label_str"] = [ID2LABEL[i] for i in pred_ids]
    # NOTE: this is max *logit*, not probability. For probs, apply softmax.
    out["pred_confidence"] = np.max(logits, axis=-1).tolist()
    return out


def eval_test(df_test_pred: pd.DataFrame, verbose: bool = False, topk: int = 10) -> None:
    y_true = np.array([df_test_pred["label_score"].values], dtype=float)
    y_score = np.array([df_test_pred["pred_label_score"].values], dtype=float)

    print("\n=== Test metrics ===")
    for k in [10, 20, 40, 100]:
        print(f"NDCG@{k}: {ndcg_score(y_true, y_score, k=k)}")

    confusion = pd.crosstab(df_test_pred["label_score"], df_test_pred["pred_label_score"])
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion)

    if verbose:
        df_sorted = df_test_pred.sort_values("pred_label_score", ascending=False).reset_index(drop=True)
        print(f"\nTop {topk} predicted sentences (by pred_label_score):")
        for i, (_, row) in enumerate(df_sorted.head(topk).iterrows(), start=1):
            print(
                f"[{i:>2}] pred={row['pred_label_str']:<16} "
                f"true={row.get('label','')} "
                f"term={row.get('term','')} | {row.get('text','')}"
            )


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_fold", default="1", help="Held-out test fold (1-6)")
    ap.add_argument("--max_length", type=int, default=256, help="Max sequence length")
    ap.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    ap.add_argument("--batch_size", type=int, default=8, help="Per-device batch size")
    ap.add_argument("--verbose", default="false", help="true|false to print top predictions")
    ap.add_argument("--cpu_only", default="true", help="true|false (set false if you have CUDA GPU)")
    # Instruction prefixing
    ap.add_argument(
        "--instruction",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help="Instruction text prepended to every example",
    )
    ap.add_argument(
        "--sep",
        type=str,
        default=DEFAULT_SEP,
        help="Separator between instruction and sentence",
    )

    args = ap.parse_args()

    test_fold = int(args.test_fold)
    verbose = (args.verbose.lower() == "true")
    cpu_only = (args.cpu_only.lower() == "true")
    instruction = args.instruction
    sep = args.sep

    print(f"Test fold: {test_fold} | Train folds: all others (5 folds)")
    print("Instruction prefixing: ON")
    print(f"Instruction: {instruction}")
    print(f"Separator: {repr(sep)}")

    train_df, test_df = build_train_test_dfs(test_fold=test_fold)

    if train_df.empty:
        raise ValueError("TRAIN df is empty — check your sources//-sentence.json paths.")
    if test_df.empty:
        raise ValueError("TEST df is empty — check your sources//-sentence.json paths.")

    output_dir = f"/content/drive/MyDrive/src/modernbert-legal-test{test_fold}"

    trainer, tokenizer = train_modernbert(
        train_df=train_df,
        max_length=args.max_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=output_dir,
        instruction=instruction,
        sep=sep,
        cpu_only=cpu_only,
    )

    df_test_pred = predict_on_df(
        trainer=trainer,
        tokenizer=tokenizer,
        df=test_df,
        max_length=args.max_length,
        instruction=instruction,
        sep=sep,
    )

    eval_test(df_test_pred, verbose=verbose, topk=10)


if __name__ == "__main__":
    main()