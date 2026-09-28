from pathlib import Path
import re
import pandas as pd

LABEL_CANDIDATES = ["label", "class", "target", "event", "category", "type", "marker"]


def read_csvs(directory):
    paths = sorted(Path(directory).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame.columns = [str(column).strip() for column in frame.columns]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False), paths


def find_target(df, requested=None):
    if requested and requested in df.columns:
        return requested
    lower = {str(column).strip().lower(): column for column in df.columns}
    for candidate in LABEL_CANDIDATES:
        if candidate in lower:
            return lower[candidate]
    raise ValueError("Target column not found. Use --target with the exact column name.")

def normalize_label(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    text = text.lower().replace("_", " ").replace("-", " ")
    compact = re.sub(r"[^a-z0-9]", "", text)

    if compact in {
        "0",
        "normal",
        "noevents",
        "noevent",
        "benign",
        "legitimate",
    }:
        return "Normal"

    if compact in {
        "1",
        "attack",
        "attacks",
        "injection",
        "intrusion",
        "malicious",
        "cyber",
    }:
        return "Attack"

    if compact in {
        "natural",
        "naturalevent",
        "naturalevents",
        "fault",
        "failure",
        "disturbance",
    }:
        return "Fault"

    if any(word in text for word in [
        "attack",
        "injection",
        "intrusion",
        "malicious",
        "cyber",
    ]):
        return "Attack"

    if any(word in text for word in [
        "fault",
        "natural event",
        "failure",
        "disturbance",
    ]):
        return "Fault"

    if any(word in text for word in [
        "normal",
        "no event",
        "benign",
        "legitimate",
    ]):
        return "Normal"

    return None

def drop_leakage_columns(X):
    remove = []
    for column in X.columns:
        normalized = re.sub(r"[^a-z0-9]", "", str(column).lower())
        if normalized in {"id", "index", "unnamed0", "sourcefile", "scenario", "scenarioid"}:
            remove.append(column)
    return X.drop(columns=remove, errors="ignore"), remove
