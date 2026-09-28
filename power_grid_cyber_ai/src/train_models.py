import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MODELS_DIR, RANDOM_STATE, RAW_DIR, RESULTS_DIR, TEST_SIZE
from src.data_loader import drop_leakage_columns, find_target, normalize_label, read_csvs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    df, paths = read_csvs(RAW_DIR)
    target = find_target(df, args.target)

    normalized = df[target].map(normalize_label)
    unknown_mask = normalized.isna()
    if unknown_mask.any():
        unknown = (
            df.loc[unknown_mask, target].astype("string").fillna("<EMPTY>")
            .value_counts(dropna=False).rename_axis("raw_label")
            .reset_index(name="count")
        )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        unknown.to_csv(RESULTS_DIR / "unknown_labels.csv", index=False)
        print("Ignored rows with missing or unknown labels:")
        print(unknown.to_string(index=False))
        print(f"Ignored rows: {int(unknown_mask.sum())}")

    df = df.loc[~unknown_mask].copy()
    y = normalized.loc[~unknown_mask]
    X, removed = drop_leakage_columns(df.drop(columns=[target]))
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().any(axis=0)]

    if y.nunique() < 3:
        raise ValueError(f"Expected 3 classes; found {sorted(y.unique().tolist())}")

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    missing_rates = X_train.isna().mean()
    feature_order = missing_rates[missing_rates <= 0.50].index.tolist()
    X_train = X_train[feature_order]
    X_test = X_test[feature_order]

    # Remove non-finite values before fitting the median imputer.
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    X_train_ready = imputer.fit_transform(X_train)
    X_test_ready = imputer.transform(X_test)
    X_train_ready = np.nan_to_num(X_train_ready, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_ready = np.nan_to_num(X_test_ready, nan=0.0, posinf=0.0, neginf=0.0)

    encoder = LabelEncoder()
    encoder.fit(["Normal", "Fault", "Attack"])
    y_train = encoder.transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)

    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_ready, y_train)

    weights = compute_sample_weight(class_weight="balanced", y=y_train)
    xgb = XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=400,
        max_depth=6, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, eval_metric="mlogloss",
        random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
    )
    xgb.fit(X_train_ready, y_train, sample_weight=weights)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dump(rf, MODELS_DIR / "random_forest.joblib")
    dump(xgb, MODELS_DIR / "xgboost_model.joblib")
    dump(imputer, MODELS_DIR / "imputer.joblib")
    dump(encoder, MODELS_DIR / "label_encoder.joblib")
    (MODELS_DIR / "feature_order.json").write_text(json.dumps(feature_order, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "target": target, "files": [p.name for p in paths],
        "removed_columns": removed, "ignored_unknown_rows": int(unknown_mask.sum()),
        "dropped_missing_columns": missing_rates[missing_rates > 0.50].index.tolist(),
        "classes": encoder.classes_.tolist(), "train_rows": len(X_train),
        "test_rows": len(X_test), "features": len(feature_order),
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({
        "y_true": encoder.inverse_transform(y_test),
        "rf_prediction": encoder.inverse_transform(rf.predict(X_test_ready)),
        "xgb_prediction": encoder.inverse_transform(xgb.predict(X_test_ready)),
    }).to_csv(RESULTS_DIR / "predictions_test.csv", index=False)
    print("Training completed successfully.")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
