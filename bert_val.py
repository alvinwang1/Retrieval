#!/usr/bin/env python3
"""
Fold-aware ModernBERT training + eval with modes:
1) instruction: instruction + sentence (single sequence)
2) qry2snt:     term + sentence (pair sequence)
3) prov2snt:    raw_provision + sentence (pair sequence)
4) spqry2snt:   raw_provision(with <term_of_interest>term</term_of_interest>) + sentence (pair sequence)
5) sp2par(snt): raw_provision + paragraph_window_around_sentence (pair sequence)
   - paragraph_window is a +/- (par_window_tokens//2) token window around the sentence inside its paragraph.
   - sentence->paragraph match is done via paragraph_id.
6) sp2sntpar:   raw_provision + (sentence [SEP] paragraph_window) (pair sequence)
   - representation: [CLS] raw_provision [SEP] sentence [SEP] paragraph_window [SEP]
   - implemented by making sequence B = "sentence <SEP> paragraph_window" and passing (A,B) to tokenizer

- saves CSVs: per_term_test_metrics.csv, test_predictions.csv
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

import torch
from torch.utils.data import Dataset
from transformers import TrainerCallback
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
    return f"{instruction}{sep}{sentence}"


# -----------------------------
# special tokens
# -----------------------------
TOI_START = "<term_of_interest>"
TOI_END = "</term_of_interest>"
SENT_START = "<sentence>"
SENT_END = "</sentence>"


def highlight_term_in_provision(raw_provision: str, term: str) -> Tuple[str, bool]:
    """
    Insert <term_of_interest> and </term_of_interest> around the first match of term in raw_provision.
    Match is case-insensitive. Preserves the original casing from the provision.

    Returns: (highlighted_text, found)
    """
    if not raw_provision or not term:
        return raw_provision, False

    pattern = re.compile(re.escape(term), flags=re.IGNORECASE)
    m = pattern.search(raw_provision)
    if not m:
        return raw_provision, False

    start, end = m.span()
    highlighted = raw_provision[:start] + TOI_START + raw_provision[start:end] + TOI_END + raw_provision[end:]
    return highlighted, True


# -----------------------------
# Paths (edit if needed)
# -----------------------------
SOURCES_ROOT = Path("/content/drive/MyDrive/src/sources")
#SOURCES_ROOT = Path("sources")
PROVISIONS_PATH = SOURCES_ROOT / "provisions.json"

# -----------------------------
# Load provisions.json once
# -----------------------------
PROVISIONS: Dict[str, Dict[str, str]] = {}
if PROVISIONS_PATH.exists():
    with PROVISIONS_PATH.open("r", encoding="utf-8") as f:
        PROVISIONS = json.load(f)
else:
    print(
        f"[warn] provisions.json not found at {PROVISIONS_PATH.resolve()}. "
        f"prov2snt/spqry2snt/sp2par(snt)/sp2sntpar will not work unless you add it."
    )


# -----------------------------
# JSON helpers
# -----------------------------
def _pick_text_field(entry: Dict[str, Any]) -> str:
    """
    Try common fields for paragraph/sentence JSON entries.
    Prefers 'text', then 'raw', then 'analyzed'.
    """
    for k in ("text", "raw", "analyzed"):
        v = entry.get(k, "")
        if isinstance(v, str) and v.strip():
            return v
    return ""


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
                "text": entry.get("text", "") or _pick_text_field(entry),
                "label": entry.get("label", ""),
                "label_score": LABEL_MAP.get(label_str, -1),
            }
        )
    return pd.DataFrame(rows)


def load_paragraph_map_for_term(term: str) -> Dict[str, str]:
    """
    Loads sources/<term_slug>/<term_slug>-paragraph.json and returns:
      paragraph_id -> paragraph_text

    Robust to slight schema differences:
      - expects each entry to have a 'paragraph_id' (or the dict key is used)
      - and some text field ('text'/'raw'/'analyzed')
    """
    term_slug = term.replace(" ", "_")
    par_path = SOURCES_ROOT / term_slug / f"{term_slug}-paragraph.json"
    if not par_path.exists():
        print(f"[warn] paragraph file not found for term={term!r}: {par_path}")
        return {}

    with par_path.open("r", encoding="utf-8") as f:
        par_json = json.load(f)

    parid2par: Dict[str, str] = {}

    # Typical format: { paragraph_id: {paragraph_id:..., text:...}, ... }
    if isinstance(par_json, dict):
        for pid_key, entry in par_json.items():
            # Determine paragraph_id:
            # 1) prefer explicit field if present
            # 2) else use the dict key
            paragraph_id = ""
            if isinstance(entry, dict):
                paragraph_id = str(entry.get("paragraph_id", "") or "").strip()
            if not paragraph_id:
                paragraph_id = str(pid_key).strip()

            if not paragraph_id:
                continue

            txt = _pick_text_field(entry) if isinstance(entry, dict) else ""
            if txt.strip():
                parid2par.setdefault(paragraph_id, txt)
    else:
        print(f"[warn] unexpected paragraph JSON structure for {par_path} (expected dict).")

    return parid2par


def load_term_df(term: str) -> pd.DataFrame:
    """
    Loads sources/<term_slug>/<term_slug>-sentence.json into a DataFrame.

    Also attaches:
      - term
      - provision_raw (from provisions.json[term]["raw"], if available)
      - provision_spqry (raw provision with <term_of_interest>term</term_of_interest> inserted)
      - paragraph_text (matched by paragraph_id from <term_slug>-paragraph.json)
        [for sp2par(snt) and sp2sntpar]
    """
    term_slug = term.replace(" ", "_")
    sent_path = SOURCES_ROOT / term_slug / f"{term_slug}-sentence.json"
    if not sent_path.exists():
        raise FileNotFoundError(f"File not found: {sent_path}")

    with sent_path.open("r", encoding="utf-8") as f:
        sent_json = json.load(f)

    df = build_records(sent_json)
    if df.empty:
        return df

    df["term"] = term

    prov = (PROVISIONS.get(term, {}) or {})
    raw = (prov.get("raw", "") or "")
    if not raw:
        print(f"[warn] provisions.json missing raw provision for term: {term}")
    df["provision_raw"] = raw

    highlighted, found = highlight_term_in_provision(raw, term)
    if raw and not found:
        print(f"[warn] spqry2snt: term not found in raw provision (term={term!r}). Using unhighlighted provision.")
    df["provision_spqry"] = highlighted

    # Paragraph mapping via paragraph_id
    parid2par = load_paragraph_map_for_term(term)
    if parid2par:
        df["paragraph_text"] = df["paragraph_id"].astype(str).map(parid2par).fillna("")
        missing_n = int((df["paragraph_text"].astype(str).str.strip() == "").sum())
        if missing_n > 0:
            print(f"[warn] sp2par/sp2sntpar: {missing_n}/{len(df)} sentences missing paragraph_text for term={term!r}")
    else:
        df["paragraph_text"] = ""

    return df


def build_fold_df(fold_idx: int) -> pd.DataFrame:
    dfs = []
    for term in folds[fold_idx]:
        df = load_term_df(term)
        if not df.empty:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def build_train_val_test_dfs(
    test_fold: int,
    val_fold: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (1 <= test_fold <= 6):
        raise ValueError(f"test_fold must be in [1,6], got {test_fold}")
    if not (1 <= val_fold <= 6):
        raise ValueError(f"val_fold must be in [1,6], got {val_fold}")
    if test_fold == val_fold:
        raise ValueError("test_fold and val_fold must be different")

    test_idx = test_fold - 1
    val_idx = val_fold - 1

    test_df = build_fold_df(test_idx)
    val_df = build_fold_df(val_idx)

    train_dfs = []
    for i in range(6):
        if i in (test_idx, val_idx):
            continue
        df = build_fold_df(i)
        if not df.empty:
            train_dfs.append(df)

    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    return train_df, val_df, test_df


# -----------------------------
# Paragraph windowing (token-based, using fast tokenizer offsets)
# -----------------------------
def clip_paragraph_around_sentence_tokens(
    paragraph: str,
    sentence: str,
    tokenizer,
    window_tokens: int,
) -> str:
    """
    Returns a substring of `paragraph` that corresponds to a token-window around the
    first occurrence of `sentence` in `paragraph`.

    Window definition:
      take sentence span in tokens, then extend by half-window on both sides:
        left = window_tokens//2, right = window_tokens - left
      final window is [sent_start-left, sent_end+right] in token indices.

    If sentence isn't found, falls back to returning the original paragraph.
    """
    paragraph = paragraph or ""
    sentence = sentence or ""
    if not paragraph.strip():
        return paragraph

    # Find sentence by character span (best effort)
    pos = paragraph.find(sentence)
    if pos < 0:
        # Try a lighter normalization if exact string doesn't match
        p2 = " ".join(paragraph.split())
        s2 = " ".join(sentence.split())
        pos2 = p2.find(s2)
        if pos2 < 0:
            return paragraph
        # If normalized matched, just return paragraph (Trainer will truncate anyway).
        return paragraph

    sent_char_start = pos
    sent_char_end = pos + len(sentence)

    # Tokenize paragraph with offsets
    enc = tokenizer(
        paragraph,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        padding=False,
    )
    offsets = enc.get("offset_mapping", None)
    if offsets is None:
        return paragraph

    # Find token span overlapping the sentence char span
    sent_tok_start = None
    sent_tok_end = None
    for i, (a, b) in enumerate(offsets):
        if a is None or b is None:
            continue
        # token overlaps [sent_char_start, sent_char_end)
        if b > sent_char_start and a < sent_char_end:
            if sent_tok_start is None:
                sent_tok_start = i
            sent_tok_end = i

    if sent_tok_start is None or sent_tok_end is None:
        return paragraph

    # Build window indices
    left = window_tokens // 2
    right = window_tokens - left
    w_start = max(0, sent_tok_start - left)
    w_end = min(len(offsets) - 1, sent_tok_end + right)

    # Convert token window -> char window using offsets
    char_start = offsets[w_start][0]
    char_end = offsets[w_end][1]
    if char_start is None or char_end is None:
        return paragraph

    return paragraph[char_start:char_end]


# -----------------------------
# Dataset
# -----------------------------
class ModernBertDataset(Dataset):
    """
    Supports:
      - instruction: single-sequence prompt
      - pair modes: context + second_text
      - sp2par(snt): context=provision_raw, second_text=paragraph_window(sentence-centered)
      - sp2sntpar: context=provision_raw, second_text=(sentence [SEP] paragraph_window)
                  => [CLS] provision [SEP] sentence [SEP] paragraph_window [SEP]

    Inputs:
      texts:     sentence text
      contexts:  term/provision/...
      paragraphs: full paragraph text (only used in sp2par(snt) and sp2sntpar)
    """
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        contexts: List[str],
        tokenizer,
        instruction: str,
        sep: str,
        max_length: int = 512,
        mode: str = "instruction",
        paragraphs: Optional[List[str]] = None,
        par_window_tokens: int = 250,
    ):
        self.texts = texts
        self.labels = labels
        self.contexts = contexts
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.sep = sep
        self.max_length = max_length
        self.mode = mode
        self.paragraphs = paragraphs
        self.par_window_tokens = par_window_tokens

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        sentence = str(self.texts[idx])
        context = str(self.contexts[idx])
        label = int(self.labels[idx])

        if self.mode == "instruction":
            prompt = build_prompt(sentence=sentence, instruction=self.instruction, sep=self.sep)
            enc = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )

        elif self.mode in ("qry2snt", "prov2snt", "spqry2snt"):
            enc = self.tokenizer(
                context,
                sentence,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )

        elif self.mode in ("sp2par(snt)", "sp2par_snt"):
            if not self.paragraphs:
                # Fallback: just use the sentence if paragraph missing
                second_text = sentence
            else:
                paragraph_full = str(self.paragraphs[idx] or "")
                if paragraph_full.strip():
                    second_text = clip_paragraph_around_sentence_tokens(
                        paragraph=paragraph_full,
                        sentence=sentence,
                        tokenizer=self.tokenizer,
                        window_tokens=self.par_window_tokens,
                    )
                else:
                    second_text = sentence

            enc = self.tokenizer(
                context,        # raw provision
                second_text,    # windowed paragraph
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )

        elif self.mode == "sp2sntpar":
            # Build paragraph window (sentence-centered). If missing, fall back gracefully.
            paragraph_window = ""
            if self.paragraphs:
                paragraph_full = str(self.paragraphs[idx] or "")
                if paragraph_full.strip():
                    paragraph_window = clip_paragraph_around_sentence_tokens(
                        paragraph=paragraph_full,
                        sentence=sentence,
                        tokenizer=self.tokenizer,
                        window_tokens=self.par_window_tokens,
                    )

            # Use tokenizer.sep_token so it becomes an actual [SEP] token id in the middle of sequence B.
            sep_tok = self.tokenizer.sep_token or "[SEP]"
            wrapped_sentence = f"{SENT_START} {sentence} {SENT_END}"
            if paragraph_window.strip():
                second_text = f"{wrapped_sentence} {sep_tok} {paragraph_window}"
            else:
                second_text = wrapped_sentence

            # Pair encoding: [CLS] context [SEP] second_text [SEP]
            # Because second_text includes sep_tok, this yields:
            # [CLS] context [SEP] sentence [SEP] paragraph_window [SEP]
            enc = self.tokenizer(
                context,        # raw provision
                second_text,    # wrapped_sentence [SEP] paragraph_window
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )

        else:
            raise ValueError(f"Unknown mode in dataset: {self.mode}")

        enc = {k: torch.tensor(v) for k, v in enc.items()}
        enc["labels"] = torch.tensor(label)
        return enc


# -----------------------------
# Metrics: NDCG on expected-value score
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    scores = np.sum(probs * np.arange(probs.shape[1]), axis=1)

    y_true = labels.reshape(1, -1).astype(float)
    y_score = scores.reshape(1, -1).astype(float)

    return {"ndcg@100": float(ndcg_score(y_true, y_score, k=100))}


# -----------------------------
# Callback to save only best model
# -----------------------------
class SaveBestModelCallback(TrainerCallback):
    def __init__(self, output_dir: str, metric_name: str = "ndcg@100"):
        self.output_dir = output_dir
        self.best_model_dir = os.path.join(output_dir, "best-model")
        self.metric_name = metric_name
        self.best_metric: Optional[float] = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        current_metric = metrics.get(f"eval_{self.metric_name}")
        if current_metric is None:
            return

        if self.best_metric is None or current_metric > self.best_metric:
            self.best_metric = current_metric
            print(f"\n[SaveBest] New best {self.metric_name}: {current_metric:.4f}")
            print(f"[SaveBest] Marking best checkpoint as {self.best_model_dir}")

            os.makedirs(self.best_model_dir, exist_ok=True)
            state.best_metric = current_metric
            state.best_model_checkpoint = self.best_model_dir


# -----------------------------
# Helpers
# -----------------------------
def _require_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} df missing columns: {missing}")


def _normalize_mode(mode: str) -> str:
    if mode == "sp2par_snt":
        return "sp2par(snt)"
    return mode


def _get_context_col(mode: str) -> str:
    if mode == "qry2snt":
        return "term"
    if mode == "prov2snt":
        return "provision_raw"
    if mode == "spqry2snt":
        return "provision_spqry"
    if mode in ("sp2par(snt)", "sp2par_snt", "sp2sntpar"):
        return "provision_raw"
    return "term"  # instruction mode (unused)


def _validate_mode_requirements(df: pd.DataFrame, mode: str, split_name: str) -> None:
    mode = _normalize_mode(mode)

    if mode in ("prov2snt", "spqry2snt", "sp2par(snt)", "sp2sntpar"):
        if not PROVISIONS:
            raise ValueError(f"{mode} mode selected but provisions.json was not loaded.")
        col = _get_context_col(mode)
        _require_columns(df, ["term", col], split_name)
        bad = df.groupby("term")[col].apply(lambda s: (s.astype(str).str.strip() == "").all())
        missing_terms = bad[bad].index.tolist()
        if missing_terms:
            raise ValueError(
                f"{mode} requires non-empty provision text for every term.\n"
                f"Missing/empty provision for {len(missing_terms)} term(s) in {split_name}: {missing_terms}"
            )

    if mode in ("sp2par(snt)", "sp2sntpar"):
        _require_columns(df, ["paragraph_id", "paragraph_text"], split_name)

        # Not fatal if some rows missing (we fallback), but warn loudly.
        missing_n = int((df["paragraph_text"].astype(str).str.strip() == "").sum())
        if missing_n > 0:
            print(
                f"[warn] {split_name}: {mode} has {missing_n}/{len(df)} rows with empty paragraph_text "
                f"(will fallback gracefully)."
            )


# -----------------------------
# Training
# -----------------------------
def train_modernbert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    mode: str,
    max_length: int,
    num_epochs: int,
    batch_size: int,
    output_dir: str,
    instruction: str,
    sep: str,
    cpu_only: bool = True,
    par_window_tokens: int = 250,
) -> Tuple[Trainer, Any]:
    mode = _normalize_mode(mode)

    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Make <term_of_interest> markers single tokens (spqry2snt)
    if mode == "spqry2snt":
        tokenizer.add_special_tokens({"additional_special_tokens": [TOI_START, TOI_END]})
    
    if mode == "sp2sntpar":
        tokenizer.add_special_tokens({"additional_special_tokens": [SENT_START, SENT_END]})

    train_labeled = train_df[train_df["label_score"] >= 0].copy()
    if train_labeled.empty:
        raise ValueError("No labeled rows in TRAIN set (label_score >= 0).")

    val_labeled = val_df[val_df["label_score"] >= 0].copy()
    if val_labeled.empty:
        raise ValueError("No labeled rows in VAL set (label_score >= 0).")

    context_col = _get_context_col(mode)

    _require_columns(train_labeled, ["text", "label_score", context_col, "term"], "TRAIN")
    _require_columns(val_labeled, ["text", "label_score", context_col, "term"], "VAL")
    _validate_mode_requirements(train_labeled, mode, "TRAIN")
    _validate_mode_requirements(val_labeled, mode, "VAL")

    train_pars = train_labeled["paragraph_text"].astype(str).tolist() if mode in ("sp2par(snt)", "sp2sntpar") else None
    val_pars = val_labeled["paragraph_text"].astype(str).tolist() if mode in ("sp2par(snt)", "sp2sntpar") else None

    train_ds = ModernBertDataset(
        texts=train_labeled["text"].tolist(),
        labels=train_labeled["label_score"].astype(int).tolist(),
        contexts=train_labeled[context_col].astype(str).tolist(),
        tokenizer=tokenizer,
        instruction=instruction,
        sep=sep,
        max_length=max_length,
        mode=mode,
        paragraphs=train_pars,
        par_window_tokens=par_window_tokens,
    )

    val_ds = ModernBertDataset(
        texts=val_labeled["text"].tolist(),
        labels=val_labeled["label_score"].astype(int).tolist(),
        contexts=val_labeled[context_col].astype(str).tolist(),
        tokenizer=tokenizer,
        instruction=instruction,
        sep=sep,
        max_length=max_length,
        mode=mode,
        paragraphs=val_pars,
        par_window_tokens=par_window_tokens,
    )

    model = ModernBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_MAP),
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
    )

    if mode == "spqry2snt" or mode == "sp2sntpar":
        model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        logging_steps=50,
        no_cuda=cpu_only,
        dataloader_pin_memory=False,
        seed=42,
        report_to=["tensorboard"],
        logging_dir=os.path.join(output_dir, "tb"),
        metric_for_best_model="ndcg@100",
        greater_is_better=True,
    )

    best_model_callback = SaveBestModelCallback(output_dir=output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        callbacks=[best_model_callback],
    )

    os.makedirs(output_dir, exist_ok=True)
    print(f"[train] output_dir = {output_dir}")
    print("[train] Evaluating every epoch, saving only best model based on val ndcg@100")

    trainer.train()

    best_dir = os.path.join(output_dir, "best-model")
    os.makedirs(best_dir, exist_ok=True)
    print(f"\n[train] Saving model+tokenizer to {best_dir}")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    return trainer, tokenizer


# -----------------------------
# Eval-only loader
# -----------------------------
def load_trainer_from_dir(
    model_dir: str,
    cpu_only: bool,
    output_dir: str,
    eval_batch_size: int,
    mode: str,
    tokenizer_source: str = "answerdotai/ModernBERT-base",
) -> Tuple[Trainer, Any]:
    mode = _normalize_mode(mode)

    if not model_dir:
        raise ValueError("model_dir is empty. Provide --model_dir when using --eval_only.")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if mode == "spqry2snt":
        tokenizer.add_special_tokens({"additional_special_tokens": [TOI_START, TOI_END]})
    
    if mode == "sp2sntpar":
        tokenizer.add_special_tokens({"additional_special_tokens": [SENT_START, SENT_END]})

    model = ModernBertForSequenceClassification.from_pretrained(model_dir)
    if mode == "spqry2snt" or mode == "sp2sntpar":
        model.resize_token_embeddings(len(tokenizer))

    eval_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=eval_batch_size,
        dataloader_pin_memory=False,
        no_cuda=cpu_only,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer, tokenizer


# -----------------------------
# Prediction + evaluation
# -----------------------------
def predict_on_df(
    trainer: Trainer,
    tokenizer,
    df: pd.DataFrame,
    mode: str,
    max_length: int,
    instruction: str,
    sep: str,
    par_window_tokens: int = 250,
) -> pd.DataFrame:
    mode = _normalize_mode(mode)

    df_lab = df[df["label_score"] >= 0].copy()
    if df_lab.empty:
        raise ValueError("No labeled rows in DF for prediction/eval.")

    context_col = _get_context_col(mode)

    _require_columns(df_lab, ["text", "label_score", context_col, "term"], "PREDICT")
    _validate_mode_requirements(df_lab, mode, "PREDICT/TEST")

    test_pars = df_lab["paragraph_text"].astype(str).tolist() if mode in ("sp2par(snt)", "sp2sntpar") else None

    ds = ModernBertDataset(
        texts=df_lab["text"].tolist(),
        labels=df_lab["label_score"].astype(int).tolist(),
        contexts=df_lab[context_col].astype(str).tolist(),
        tokenizer=tokenizer,
        instruction=instruction,
        sep=sep,
        max_length=max_length,
        mode=mode,
        paragraphs=test_pars,
        par_window_tokens=par_window_tokens,
    )

    preds = trainer.predict(ds)
    logits = preds.predictions

    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    expected_score = np.sum(probs * np.arange(probs.shape[1]), axis=1)
    pred_ids = np.argmax(logits, axis=-1)

    out = df_lab.reset_index(drop=True)
    out["pred_label_score"] = pred_ids
    out["pred_label_str"] = [ID2LABEL[int(i)] for i in pred_ids]
    out["pred_expected_score"] = expected_score.tolist()
    out["pred_confidence"] = np.max(probs, axis=-1).tolist()
    return out


def eval_test(
    df_test_pred: pd.DataFrame,
    verbose: bool = False,
    topk: int = 10,
    k_list: List[int] = [10, 20, 40, 100],
    macro_k: int = 100,
    min_term_n: int = 2,
) -> Dict[str, Any]:
    y_true = np.array([df_test_pred["label_score"].values], dtype=float)
    y_score = np.array([df_test_pred["pred_expected_score"].values], dtype=float)

    pooled = {}
    print("\n=== Test metrics (pooled over all terms) ===")
    for k in k_list:
        pooled[f"ndcg@{k}"] = float(ndcg_score(y_true, y_score, k=k))
        print(f"NDCG@{k}: {pooled[f'ndcg@{k}']:.4f}")

    confusion = pd.crosstab(df_test_pred["label_score"], df_test_pred["pred_label_score"])
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion)

    term_rows = []
    for term, g in df_test_pred.groupby("term"):
        if len(g) < min_term_n:
            continue
        yt = np.array([g["label_score"].values], dtype=float)
        ys = np.array([g["pred_expected_score"].values], dtype=float)

        row = {"term": term, "n": int(len(g))}
        for k in k_list:
            row[f"ndcg@{k}"] = float(ndcg_score(yt, ys, k=k))
        term_rows.append(row)

    per_term_df = pd.DataFrame(term_rows)
    if not per_term_df.empty:
        per_term_df = per_term_df.sort_values(f"ndcg@{macro_k}", ascending=False).reset_index(drop=True)
        macro_avg = float(per_term_df[f"ndcg@{macro_k}"].mean())
    else:
        macro_avg = float("nan")

    print("\n=== Per-term NDCG (macro averaged) ===")
    if np.isnan(macro_avg):
        print(f"[warn] No per-term metrics computed (min_term_n={min_term_n}?).")
    else:
        print(f"Macro-avg NDCG@{macro_k}: {macro_avg:.4f} "
              f"(over {len(per_term_df)} terms, min_term_n={min_term_n})")

        cols = ["term", "n"] + [f"ndcg@{k}" for k in k_list]
        print("\nPer-term summary (top/bottom 5 by ndcg@100):")
        print(per_term_df[cols].head(5).to_string(index=False))
        print("...")
        print(per_term_df[cols].tail(5).to_string(index=False))

    if verbose:
        df_sorted = df_test_pred.sort_values("pred_expected_score", ascending=False).reset_index(drop=True)
        print(f"\nTop {topk} predicted sentences (by expected score):")
        for i, (_, row) in enumerate(df_sorted.head(topk).iterrows(), start=1):
            print(
                f"[{i:>2}] score={row['pred_expected_score']:.3f} "
                f"pred={row['pred_label_str']:<16} "
                f"true={row.get('label','')} "
                f"term={row.get('term','')} | {row.get('text','')}"
            )

    return {
        "pooled": pooled,
        "macro_avg_ndcg@100": macro_avg,
        "per_term": per_term_df,
        "confusion": confusion,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_fold", default="1", help="Held-out test fold (1-6)")
    ap.add_argument("--val_fold", default="2", help="Validation fold (1-6)")
    ap.add_argument("--max_length", type=int, default=256, help="Max sequence length")
    ap.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    ap.add_argument("--batch_size", type=int, default=8, help="Per-device batch size")
    ap.add_argument("--verbose", default="false", help="true|false to print top predictions")
    ap.add_argument("--cpu_only", default="true", help="true|false (set false if you have CUDA GPU)")

    ap.add_argument(
        "--mode",
        type=str,
        default="instruction",
        choices=[
            "instruction",
            "qry2snt",
            "prov2snt",
            "spqry2snt",
            "sp2par(snt)",
            "sp2par_snt",
            "sp2sntpar",
        ],
        help=(
            "instruction: instruction+sentence; "
            "qry2snt: term+sentence; "
            "prov2snt: raw_provision+sentence; "
            "spqry2snt: provision(with highlighted term)+sentence; "
            "sp2par(snt): raw_provision + paragraph_window_around_sentence; "
            "sp2sntpar: raw_provision + (sentence [SEP] paragraph_window)"
        ),
    )

    ap.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    ap.add_argument("--sep", type=str, default=DEFAULT_SEP)

    ap.add_argument("--eval_only", default="false", help="true|false. If true, skip training and only evaluate a saved model.")
    ap.add_argument("--model_dir", type=str, default="", help="Path to a saved model dir (e.g., .../best-model). Required if --eval_only true.")

    # paragraph window size in tokens (total window around sentence)
    ap.add_argument("--par_window_tokens", type=int, default=250, help="Total token window around sentence when mode uses paragraph windows (sp2par(snt)/sp2sntpar).")

    args = ap.parse_args()

    test_fold = int(args.test_fold)
    val_fold = int(args.val_fold)
    verbose = (args.verbose.lower() == "true")
    cpu_only = (args.cpu_only.lower() == "true")

    mode = _normalize_mode(args.mode)
    instruction = args.instruction
    sep = args.sep
    eval_only = (args.eval_only.lower() == "true")
    model_dir = args.model_dir
    par_window_tokens = int(args.par_window_tokens)

    print(f"TEST fold: {test_fold} | VAL fold: {val_fold} | TRAIN folds: all others (4 folds)")
    print(f"Mode: {mode}")

    if mode == "qry2snt":
        print("Using qry2snt: [CLS] term [SEP] sentence [SEP]")
    elif mode == "prov2snt":
        print("Using prov2snt: [CLS] raw_provision [SEP] sentence [SEP]")
    elif mode == "spqry2snt":
        print(f"Using spqry2snt: [CLS] raw_provision(with {TOI_START}term{TOI_END}) [SEP] sentence [SEP]")
    elif mode == "sp2par(snt)":
        print(f"Using sp2par(snt): [CLS] raw_provision [SEP] paragraph_window({par_window_tokens} tok) [SEP]")
        print("Paragraphs are matched by paragraph_id from <term_slug>-paragraph.json")
    elif mode == "sp2sntpar":
        print(f"Using sp2sntpar: [CLS] raw_provision [SEP] sentence [SEP] paragraph_window({par_window_tokens} tok) [SEP]")
        print("Paragraphs are matched by paragraph_id from <term_slug>-paragraph.json")
    else:
        print("Using instruction: (instruction + sep + sentence) (sentence-only if instruction/sep empty)")
        print(f"Instruction: {instruction}")
        print(f"Separator: {repr(sep)}")

    if eval_only:
        print("EVAL-ONLY: will load saved weights and skip training.")
        if not model_dir:
            raise ValueError("--model_dir is required when --eval_only true")

    train_df, val_df, test_df = build_train_val_test_dfs(test_fold=test_fold, val_fold=val_fold)
    if test_df.empty:
        raise ValueError("TEST df is empty — check your sources paths.")
    output_dir = f"/content/drive/MyDrive/src/modernbert-legal-{mode}-test{test_fold}-val{val_fold}"
    #output_dir = f"modernbert-legal-{mode}-test{test_fold}-val{val_fold}"
    os.makedirs(output_dir, exist_ok=True)

    if eval_only:
        trainer, tokenizer = load_trainer_from_dir(
            model_dir=model_dir,
            cpu_only=cpu_only,
            output_dir=os.path.join(output_dir, "eval_tmp"),
            eval_batch_size=args.batch_size,
            mode=mode,
        )
    else:
        if train_df.empty:
            raise ValueError("TRAIN df is empty — check your sources paths.")
        if val_df.empty:
            raise ValueError("VAL df is empty — check your sources paths.")

        trainer, tokenizer = train_modernbert(
            train_df=train_df,
            val_df=val_df,
            mode=mode,
            max_length=args.max_length,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=output_dir,
            instruction=instruction,
            sep=sep,
            cpu_only=cpu_only,
            par_window_tokens=par_window_tokens,
        )

    df_test_pred = predict_on_df(
        trainer=trainer,
        tokenizer=tokenizer,
        df=test_df,
        mode=mode,
        max_length=args.max_length,
        instruction=instruction,
        sep=sep,
        par_window_tokens=par_window_tokens,
    )

    results = eval_test(
        df_test_pred,
        verbose=verbose,
        topk=10,
        macro_k=100,
        min_term_n=2,
    )

    per_term_path = os.path.join(output_dir, "per_term_test_metrics.csv")
    results["per_term"].to_csv(per_term_path, index=False)
    print(f"\nSaved per-term metrics to: {per_term_path}")

    preds_path = os.path.join(output_dir, "test_predictions.csv")
    df_test_pred.to_csv(preds_path, index=False)
    print(f"Saved test predictions to: {preds_path}")


if __name__ == "__main__":
    main()
