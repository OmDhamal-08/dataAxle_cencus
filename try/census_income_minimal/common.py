import argparse
import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_TRAINING_CSV = Path(r"C:\Users\OM DHAMAL\Downloads\census_data_raw.csv")
MODEL_DIR = Path(__file__).resolve().parent / "models"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education_level",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
    "random_flag",
    "source_system",
]

RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["event_time_std"]

DERIVED_NUMERIC_FEATURES = [
    "capital_net",
    "capital_total",
    "has_capital_gain",
    "has_capital_loss",
    "hours_x_education",
    "age_x_education",
    "event_year",
    "event_month",
    "event_dayofweek",
    "event_hour",
]

DERIVED_CATEGORICAL_FEATURES = [
    "age_bucket",
    "hours_bucket",
    "education_group",
    "marital_group",
    "capital_profile",
    "is_us_native",
    "event_quarter",
    "event_is_weekend",
]

MODEL_NUMERIC_FEATURES = NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES
MODEL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + DERIVED_CATEGORICAL_FEATURES

FIELD_ALIASES = {
    "education": "education_level",
    "education_level": "education_level",
    "education_num": "education_num",
    "marital_status": "marital_status",
    "occupation": "occupation",
    "relationship": "relationship",
    "capital_gain": "capital_gain",
    "capital_loss": "capital_loss",
    "hours_per_week": "hours_per_week",
    "native_country": "native_country",
    "event_time": "event_time_std",
    "event_time_std": "event_time_std",
    "random_flag": "random_flag",
    "source_system": "source_system",
}


def standardize_column_name(name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return FIELD_ALIASES.get(text, text)


def blank_to_none(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"null", "none", "nan", "?"}:
        return None
    return text


def clean_number(value):
    value = blank_to_none(value)
    if value is None:
        return np.nan

    text = str(value).strip().lower().replace(",", "")
    if text.startswith("error"):
        return np.nan

    multiplier = 1000.0 if text.endswith("k") else 1.0
    number = re.sub(r"[^0-9.-]", "", text)
    if not re.fullmatch(r"-?[0-9]+(\.[0-9]+)?", number):
        return np.nan
    return float(number) * multiplier


def clean_age(value):
    value = blank_to_none(value)
    if value is None:
        return np.nan
    match = re.search(r"([0-9]+)", str(value))
    return float(match.group(1)) if match else np.nan


def clean_category(value):
    value = blank_to_none(value)
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    return "unknown" if text.startswith("error") else text


def normalize_income(value):
    value = blank_to_none(value)
    if value is None:
        return None

    text = re.sub(r"\s+", "", str(value).strip().lower())
    if text in {"<=80k", "=<80k", "le80k", "lessorequal80k", "<=50k", "low", "0"}:
        return "<=80k"
    if text in {">80k", "gt80k", "greaterthan80k", ">50k", "high", "1"}:
        return ">80k"
    return None


def parse_event_time(value):
    value = blank_to_none(value)
    if value is None:
        return pd.NaT

    text = str(value).strip()
    if re.fullmatch(r"[0-9]{10}", text):
        return pd.to_datetime(int(text), unit="s", errors="coerce", utc=True).tz_localize(None)
    if re.fullmatch(r"[0-9]{13}", text):
        return pd.to_datetime(int(text), unit="ms", errors="coerce", utc=True).tz_localize(None)

    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if not pd.isna(parsed):
        return parsed.tz_localize(None).to_pydatetime()
    return pd.NaT


def normalize_input_frame(records) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        df = records.copy()
    elif isinstance(records, dict):
        df = pd.DataFrame([records])
    else:
        df = pd.DataFrame(records)

    df.columns = [standardize_column_name(column) for column in df.columns]
    normalized = pd.DataFrame(index=df.index)

    for column in NUMERIC_FEATURES:
        source = df[column] if column in df else pd.Series([None] * len(df), index=df.index)
        normalized[column] = source.map(clean_age if column == "age" else clean_number)

    for column in CATEGORICAL_FEATURES:
        source = df[column] if column in df else pd.Series([None] * len(df), index=df.index)
        normalized[column] = source.map(clean_category)

    if "event_time_std" in df:
        normalized["event_time_std"] = df["event_time_std"].map(parse_event_time)
    else:
        normalized["event_time_std"] = pd.NaT

    return normalized[RAW_FEATURES]


class CensusFeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = normalize_input_frame(X)
        engineered = pd.DataFrame(index=df.index)

        for column in NUMERIC_FEATURES:
            engineered[column] = pd.to_numeric(df[column], errors="coerce")
        for column in CATEGORICAL_FEATURES:
            engineered[column] = df[column].map(clean_category)

        event_time = pd.to_datetime(df["event_time_std"].map(parse_event_time), errors="coerce")
        age = engineered["age"]
        education_num = engineered["education_num"]
        capital_gain = engineered["capital_gain"].fillna(0)
        capital_loss = engineered["capital_loss"].fillna(0)
        hours = engineered["hours_per_week"]

        engineered["capital_net"] = capital_gain - capital_loss
        engineered["capital_total"] = capital_gain + capital_loss
        engineered["has_capital_gain"] = (capital_gain > 0).astype(int)
        engineered["has_capital_loss"] = (capital_loss > 0).astype(int)
        engineered["hours_x_education"] = hours * education_num
        engineered["age_x_education"] = age * education_num
        engineered["event_year"] = event_time.dt.year
        engineered["event_month"] = event_time.dt.month
        engineered["event_dayofweek"] = event_time.dt.dayofweek
        engineered["event_hour"] = event_time.dt.hour

        engineered["age_bucket"] = pd.cut(
            age,
            bins=[0, 24, 34, 44, 54, 64, np.inf],
            labels=["<=24", "25-34", "35-44", "45-54", "55-64", "65+"],
        ).astype("object").fillna("unknown")
        engineered["hours_bucket"] = pd.cut(
            hours,
            bins=[0, 19, 34, 40, 50, np.inf],
            labels=["part_time", "reduced", "standard", "extended", "heavy"],
        ).astype("object").fillna("unknown")
        engineered["education_group"] = pd.cut(
            education_num,
            bins=[0, 8, 12, 13, 14, np.inf],
            labels=["pre_hs", "hs_or_some_college", "bachelors", "masters", "advanced"],
        ).astype("object").fillna("unknown")

        marital = engineered["marital_status"].astype(str)
        engineered["marital_group"] = np.select(
            [
                marital.str.contains("married", regex=False),
                marital.str.contains("never", regex=False),
                marital.str.contains("divorced", regex=False),
                marital.str.contains("separated", regex=False),
                marital.str.contains("widowed", regex=False),
            ],
            ["married", "never_married", "divorced", "separated", "widowed"],
            default="unknown",
        )
        engineered["capital_profile"] = np.select(
            [(capital_gain > 0) & (capital_loss > 0), capital_gain > 0, capital_loss > 0],
            ["both", "gain", "loss"],
            default="none",
        )
        country_key = engineered["native_country"].str.replace(r"[^0-9a-z]+", "", regex=True)
        engineered["is_us_native"] = np.where(country_key.eq("unitedstates"), "yes", "no")
        engineered["event_quarter"] = event_time.dt.quarter.map(
            lambda value: f"q{int(value)}" if pd.notna(value) else "unknown"
        )
        engineered["event_is_weekend"] = np.where(
            event_time.dt.dayofweek.isna(),
            "unknown",
            np.where(event_time.dt.dayofweek >= 5, "yes", "no"),
        )

        return engineered[MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES]


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), MODEL_NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                MODEL_CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_model_pipeline(classifier, scale_numeric: bool) -> Pipeline:
    return Pipeline(
        [
            ("features", CensusFeatureBuilder()),
            ("preprocessor", make_preprocessor(scale_numeric=scale_numeric)),
            ("classifier", classifier),
        ]
    )


def load_training_data(path: Path):
    raw_df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    raw_df.columns = [standardize_column_name(column) for column in raw_df.columns]

    if "income" not in raw_df.columns:
        raise ValueError("Training CSV must include an income column.")

    X = normalize_input_frame(raw_df)
    y = raw_df["income"].map(normalize_income)
    keep = y.isin(["<=80k", ">80k"])
    return X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)


def evaluate_classifier(pipeline, X_test, y_test, label_encoder=None):
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)

    if label_encoder is not None:
        y_for_metrics = label_encoder.transform(y_test)
        positive_index = int(np.where(label_encoder.classes_ == ">80k")[0][0])
        positive_scores = proba[:, positive_index]
    else:
        y_for_metrics = y_test
        positive_index = list(pipeline.classes_).index(">80k")
        positive_scores = proba[:, positive_index]

    return {
        "accuracy": round(float(accuracy_score(y_for_metrics, pred)), 4),
        "precision": round(float(precision_score(y_for_metrics, pred, average="weighted")), 4),
        "recall": round(float(recall_score(y_for_metrics, pred, average="weighted")), 4),
        "f1_score": round(float(f1_score(y_for_metrics, pred, average="weighted")), 4),
        "roc_auc": round(float(roc_auc_score(y_for_metrics, positive_scores)), 4),
    }


def save_model_bundle(path: Path, model_name: str, pipeline, metrics, label_encoder=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_name": model_name,
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "features": RAW_FEATURES,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    with path.open("wb") as file:
        pickle.dump(bundle, file)


def load_model_bundle(path: Path):
    path = Path(path)
    with path.open("rb") as file:
        return pickle.load(file)


def positive_probability(bundle, probabilities):
    label_encoder = bundle.get("label_encoder")
    if label_encoder is not None:
        positive_index = int(np.where(label_encoder.classes_ == ">80k")[0][0])
    else:
        positive_index = list(bundle["pipeline"].classes_).index(">80k")
    return probabilities[:, positive_index]


def decode_predictions(bundle, predictions):
    label_encoder = bundle.get("label_encoder")
    if label_encoder is None:
        return predictions
    return label_encoder.inverse_transform(predictions)


def training_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", type=Path, default=DEFAULT_TRAINING_CSV, help="Training CSV path")
    parser.add_argument("--models-dir", type=Path, default=MODEL_DIR, help="Folder for pickle models")
    return parser


def print_training_result(model_name: str, model_path: Path, metrics: dict):
    print(json.dumps({"model": model_name, "model_path": str(model_path), "metrics": metrics}, indent=2))
