# phase3/security/ids_preprocessor.py
"""
CIC-IDS2017 Dataset Preprocessor for Phase 3.

Loads 8 parquet files, inspects schema, cleans data, performs
a day-based train/val/test split (no data leakage), scales features,
and saves preprocessing artifacts.

Label mapping:
    BENIGN            → 0 (Normal)
    All attack labels → 1 (Anomalous/Malicious)

Split strategy (day-based, no leakage):
    - Monday (Benign only)         → training pool
    - Tuesday-Friday (attack days) → 60% train / 20% val / 20% test
    - Final train = Monday + 60% attack rows
    - Final val   = 20% attack rows
    - Final test  = 20% attack rows

Fixed random seed: 42
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
IDS_DIR   = Path("D:/road obstacle detection/CIS-IDS 2017")
OUT_DIR   = Path("D:/road obstacle detection/road_damage_research/results/phase3")
CKPT_DIR  = Path("D:/road obstacle detection/road_damage_research/phase3/checkpoints")
SEED      = 42

# ── File → day mapping ────────────────────────────────────────────────────────
FILE_DAY = {
    "Benign-Monday-no-metadata.parquet":         "Monday",
    "Bruteforce-Tuesday-no-metadata.parquet":    "Tuesday",
    "DoS-Wednesday-no-metadata.parquet":         "Wednesday",
    "Infiltration-Thursday-no-metadata.parquet": "Thursday",
    "WebAttacks-Thursday-no-metadata.parquet":   "Thursday",
    "Botnet-Friday-no-metadata.parquet":         "Friday",
    "DDoS-Friday-no-metadata.parquet":           "Friday",
    "Portscan-Friday-no-metadata.parquet":       "Friday",
}

# Non-feature columns to drop
DROP_COLS = {"timestamp", "Timestamp", "flow_id", "Flow ID",
             "source_ip", "Source IP", "src_ip",
             "destination_ip", "Destination IP", "dst_ip",
             "src_port", "dst_port"}

LABEL_COL_CANDIDATES = ["label", "Label", "class", "Class", "attack_type"]

def _find_label_col(df: pd.DataFrame) -> str:
    for c in LABEL_COL_CANDIDATES:
        if c in df.columns:
            return c
    # Try case-insensitive
    low = {c.lower(): c for c in df.columns}
    for cand in LABEL_COL_CANDIDATES:
        if cand.lower() in low:
            return low[cand.lower()]
    raise ValueError(f"Cannot find label column. Columns: {list(df.columns)}")


def load_and_inspect():
    """Load all parquet files, print schema summary."""
    files = sorted(IDS_DIR.glob("*.parquet"))
    print(f"Found {len(files)} parquet files.")
    frames = []
    label_col = None

    for f in files:
        print(f"\n  Loading: {f.name}")
        df = pd.read_parquet(f)
        print(f"    Shape: {df.shape}")

        if label_col is None:
            label_col = _find_label_col(df)
            print(f"    Label column: '{label_col}'")
            print(f"    Columns ({len(df.columns)}): {list(df.columns)[:10]} ...")

        # Attach day
        day = FILE_DAY.get(f.name, "Unknown")
        df["_day"] = day
        df["_file"] = f.name

        # Value counts
        vc = df[label_col].value_counts()
        print(f"    Label distribution:\n{vc.to_string()}")

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined shape: {combined.shape}")
    return combined, label_col


def clean_and_encode(df: pd.DataFrame, label_col: str):
    """Clean, encode labels, drop metadata columns."""
    print("\n=== Cleaning ===")

    # ── Normalize column names ────────────────────────────────────────────────
    df.columns = [c.strip() for c in df.columns]

    # ── Drop metadata columns ─────────────────────────────────────────────────
    drop = [c for c in df.columns if c in DROP_COLS or c.strip() in DROP_COLS]
    # Keep _day and _file for split
    df = df.drop(columns=drop, errors="ignore")
    print(f"  Dropped metadata columns: {drop}")

    # ── Label encoding ────────────────────────────────────────────────────────
    raw_labels = df[label_col].astype(str).str.strip()
    label_map_raw = {}
    for lbl in sorted(raw_labels.unique()):
        is_benign = lbl.upper() in ("BENIGN", "NORMAL", "0")
        label_map_raw[lbl] = 0 if is_benign else 1
    df["y"] = raw_labels.map(label_map_raw)
    df = df.drop(columns=[label_col])
    print(f"  Label mapping: {label_map_raw}")
    print(f"  Class distribution after encoding:\n{df['y'].value_counts().to_string()}")

    # ── Drop string/object columns (except internal) ──────────────────────────
    obj_cols = [c for c in df.columns
                if df[c].dtype == object and c not in ("_day", "_file", "y")]
    if obj_cols:
        print(f"  Dropping non-numeric object columns: {obj_cols}")
        df = df.drop(columns=obj_cols)

    # ── Replace Inf ───────────────────────────────────────────────────────────
    num_cols = [c for c in df.columns if c not in ("_day", "_file", "y")]
    before_inf = (df[num_cols] == np.inf).sum().sum() + (df[num_cols] == -np.inf).sum().sum()
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    print(f"  Infinite values replaced: {before_inf}")

    # ── Drop NaN ─────────────────────────────────────────────────────────────
    before_nan = df[num_cols].isna().sum().sum()
    df = df.dropna(subset=num_cols)
    print(f"  NaN rows dropped: {before_nan} values → {df.shape[0]} rows remain")

    # ── Duplicate removal ─────────────────────────────────────────────────────
    feat_cols = num_cols + ["y"]
    feat_cols = [c for c in feat_cols if c in df.columns]
    n_before = len(df)
    df = df.drop_duplicates(subset=feat_cols)
    print(f"  Duplicates removed: {n_before - len(df)} ({len(df)} remain)")

    return df, label_map_raw


def split_data(df: pd.DataFrame):
    """
    Day-based split:
      - Monday rows (all benign) → training
      - Attack days → 60/20/20 stratified split per day
    """
    print("\n=== Splitting ===")
    rng = np.random.default_rng(SEED)

    monday = df[df["_day"] == "Monday"].copy()
    attack = df[df["_day"] != "Monday"].copy()

    print(f"  Monday (benign) rows: {len(monday)}")
    print(f"  Attack-day rows: {len(attack)}")

    # Stratified split of attack days
    from sklearn.model_selection import train_test_split
    atk_train, atk_temp = train_test_split(
        attack, test_size=0.40, random_state=SEED, stratify=attack["y"])
    atk_val, atk_test = train_test_split(
        atk_temp, test_size=0.50, random_state=SEED, stratify=atk_temp["y"])

    train = pd.concat([monday, atk_train], ignore_index=True)
    val   = atk_val.reset_index(drop=True)
    test  = atk_test.reset_index(drop=True)

    print(f"  Train: {len(train)}  |  Val: {len(val)}  |  Test: {len(test)}")
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        vc = split["y"].value_counts()
        print(f"  {name} class dist: {vc.to_dict()}")

    return train, val, test


def scale_features(train, val, test):
    """Fit StandardScaler on train, transform all."""
    print("\n=== Scaling ===")
    meta_cols = ["_day", "_file", "y"]
    feat_cols = [c for c in train.columns if c not in meta_cols]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feat_cols].values.astype(np.float32))
    X_val   = scaler.transform(val[feat_cols].values.astype(np.float32))
    X_test  = scaler.transform(test[feat_cols].values.astype(np.float32))

    y_train = train["y"].values.astype(np.int32)
    y_val   = val["y"].values.astype(np.int32)
    y_test  = test["y"].values.astype(np.int32)

    print(f"  Features: {len(feat_cols)}")
    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feat_cols, scaler


def save_artifacts(scaler, feat_cols, label_map, split_info):
    """Save preprocessing artifacts."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    import joblib
    joblib.dump(scaler, CKPT_DIR / "ids_scaler.joblib")
    print(f"  Saved scaler → {CKPT_DIR / 'ids_scaler.joblib'}")

    meta = {
        "feature_columns": feat_cols,
        "label_mapping":   {k: int(v) for k, v in label_map.items()},
        "split_strategy":  "day-based: Monday=train, attacks=60/20/20",
        "random_seed":     SEED,
        "n_features":      len(feat_cols),
        "split_sizes":     split_info,
    }
    with open(CKPT_DIR / "ids_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata → {CKPT_DIR / 'ids_metadata.json'}")

    np.save(str(CKPT_DIR / "ids_feature_names.npy"), np.array(feat_cols))


def run_preprocessing():
    """Full preprocessing pipeline. Returns (X_train, y_train, X_val, y_val, X_test, y_test, feat_cols, scaler)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    df, label_col = load_and_inspect()
    df, label_map = clean_and_encode(df, label_col)
    train, val, test = split_data(df)

    X_tr, y_tr, X_val, y_val, X_te, y_te, feat_cols, scaler = scale_features(train, val, test)

    split_info = {
        "train": int(len(X_tr)), "val": int(len(X_val)), "test": int(len(X_te))
    }
    save_artifacts(scaler, feat_cols, label_map, split_info)

    print("\n=== Preprocessing complete ===")
    return X_tr, y_tr, X_val, y_val, X_te, y_te, feat_cols, scaler


if __name__ == "__main__":
    run_preprocessing()
