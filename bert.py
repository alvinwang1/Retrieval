import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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
# Data helpers
# -----------------------------
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


class ModernBertDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        enc = {k: torch.tensor(v) for k, v in enc.items()}
        enc["labels"] = torch.tensor(label)
        return enc


def train_and_eval_modernbert(
    df: pd.DataFrame,
    max_length: int = 512,
    num_epochs: int = 3,
    batch_size: int = 8,
) -> pd.DataFrame:
    """
    Train ModernBERT-base on df['text'], df['label_score'] (0-3),
    then predict labels for all rows. Returns df with
    'pred_label_str', 'pred_label_score', 'pred_confidence'.
    """
    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use only rows with a valid human label to train
    df_labeled = df[df["label_score"] >= 0].copy()
    texts = df_labeled["text"].tolist()
    labels = df_labeled["label_score"].astype(int).tolist()

    if len(df_labeled) == 0:
        raise ValueError("No labeled rows (label_score >= 0) for ModernBERT training.")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_ds = ModernBertDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    val_ds = ModernBertDataset(val_texts, val_labels, tokenizer, max_length=max_length)

    model = ModernBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_MAP),
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
    )

    training_args = TrainingArguments(
        output_dir="./modernbert-legal",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        no_cuda=True,                # CPU only (good for your M1 Air)
        dataloader_pin_memory=False, # avoid pin_memory warning on CPU
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=True)


    # Inference on ALL rows (including unlabeled) for ranking/eval
    all_texts = df["text"].tolist()
    model.eval()
    pred_label_ids = []
    pred_scores = []

    with torch.no_grad():
        for i in range(0, len(all_texts), 32):
            batch_texts = all_texts[i:i+32]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            outputs = model(**enc)
            logits = outputs.logits
            batch_ids = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            batch_conf = torch.max(logits, dim=-1).values.cpu().numpy().tolist()
            pred_label_ids.extend(batch_ids)
            pred_scores.extend(batch_conf)

    pred_label_strs = [ID2LABEL[i] for i in pred_label_ids]

    df_out = df.copy()
    df_out["pred_label_str"] = pred_label_strs
    df_out["pred_label_score"] = pred_label_ids          # predicted class 0–3
    df_out["pred_confidence"] = pred_scores              # max logit (for ranking)

    return df_out


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sentences",
        default="sources/independent_economic_value/independent_economic_value-sentence.json",
        help="Path to sentences json",
    )
    ap.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length for ModernBERT",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device batch size",
    )
    ap.add_argument(
        "--verbose",
        default="false",
        help="Verbose output (top-k predictions)",
    )
    args = ap.parse_args()

    sent_path = Path(args.sentences)
    if not sent_path.exists():
        raise FileNotFoundError(f"File not found: {sent_path}")

    with sent_path.open("r", encoding="utf-8") as f:
        sent_json = json.load(f)

    df = build_records(sent_json)
    if df.empty:
        raise ValueError("No sentences loaded.")

    # Train + predict with ModernBERT
    df_pred = train_and_eval_modernbert(
        df,
        max_length=args.max_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Only evaluate where we have ground-truth label_score
    df_eval = df_pred[df_pred["label_score"] >= 0].copy()

    # Use predicted class id as ranking score for NDCG (or you can use pred_confidence)
    y_true = np.array([df_eval["label_score"].values], dtype=float)
    y_score = np.array([df_eval["pred_label_score"].values], dtype=float)

    ndcg_scores = []
    for k in [10, 20, 40, 100]:
        ndcg = ndcg_score(y_true, y_score, k=k)
        ndcg_scores.append(ndcg)

    print("Method: modernbert | Model: answerdotai/ModernBERT-base")
    for k, ndcg in zip([10, 20, 40, 100], ndcg_scores):
        print(f"NDCG@{k}: {ndcg}")

    # Confusion matrix (true vs predicted labels)
    confusion = pd.crosstab(df_eval["label_score"], df_eval["pred_label_score"])
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion)

    # Optional: show top-k sentences by predicted "relevance" (here: predicted label_score or confidence)
    topk = 10
    if args.verbose == "true":
        df_sorted = df_eval.sort_values("pred_label_score", ascending=False).reset_index(drop=True)
        print("\nTop predicted sentences:")
        for i, (_, row) in enumerate(df_sorted.head(topk).iterrows(), start=1):
            print(f"[{i:>2}] pred={row['pred_label_str']:<16} true={row['label']} | {row['text']}")


if __name__ == "__main__":
    main()
