import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 67
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
SPLIT_PATHS = {
    "train": DATA_DIR / "nyt_train.csv",
    "valid": DATA_DIR / "nyt_valid.csv",
    "test": DATA_DIR / "nyt_test.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Task 1 Binary BoW and Word Frequency experiments."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_DIR / "nyt.csv",
        help="Path to the complete NYT CSV file.",
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        help="Regenerate the 80/10/10 stratified data split.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Optional vocabulary limit. The default uses the full training vocabulary.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=1,
        help="Minimum training-document frequency for a token (default: 1).",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for Logistic Regression.",
    )
    return parser.parse_args()


def validate_dataframe(dataframe: pd.DataFrame, source: Path) -> pd.DataFrame:
    required_columns = {"text", "label"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{source} is missing required columns: {missing}")

    cleaned = dataframe.loc[:, ["text", "label"]].dropna().copy()
    cleaned["text"] = cleaned["text"].astype(str)
    cleaned["label"] = cleaned["label"].astype(str)
    if cleaned.empty:
        raise ValueError(f"{source} contains no usable rows")
    return cleaned


def create_splits(data_path: Path) -> dict[str, pd.DataFrame]:
    dataframe = validate_dataframe(pd.read_csv(data_path), data_path)
    train, remainder = train_test_split(
        dataframe,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=dataframe["label"],
    )
    valid, test = train_test_split(
        remainder,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=remainder["label"],
    )

    splits = {
        "train": train.reset_index(drop=True),
        "valid": valid.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split_dataframe in splits.items():
        split_dataframe.to_csv(SPLIT_PATHS[split_name], index=False)
    return splits


def load_splits(data_path: Path, resplit: bool) -> dict[str, pd.DataFrame]:
    if resplit or not all(path.exists() for path in SPLIT_PATHS.values()):
        return create_splits(data_path)

    return {
        split_name: validate_dataframe(pd.read_csv(path), path)
        for split_name, path in SPLIT_PATHS.items()
    }


def make_binary(matrix: csr_matrix) -> csr_matrix:
    binary_matrix = matrix.copy()
    binary_matrix.data = np.ones_like(binary_matrix.data)
    return binary_matrix


def evaluate(
    model: LogisticRegression,
    features: csr_matrix,
    labels: pd.Series,
) -> tuple[dict[str, float], np.ndarray]:
    predictions = model.predict(features)
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }
    return metrics, predictions


def run_experiment(
    method_name: str,
    train_features: csr_matrix,
    valid_features: csr_matrix,
    test_features: csr_matrix,
    splits: dict[str, pd.DataFrame],
    regularization_c: float,
) -> list[dict[str, str | float | int]]:
    model = LogisticRegression(
        C=regularization_c,
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    model.fit(train_features, splits["train"]["label"])

    rows = []
    for split_name, features in (
        ("validation", valid_features),
        ("test", test_features),
    ):
        dataframe_key = "valid" if split_name == "validation" else "test"
        metrics, predictions = evaluate(
            model,
            features,
            splits[dataframe_key]["label"],
        )
        rows.append(
            {
                "method": method_name,
                "split": split_name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "vocabulary_size": train_features.shape[1],
            }
        )

        prediction_output = splits[dataframe_key].copy()
        prediction_output["prediction"] = predictions
        prediction_output.to_csv(
            OUTPUT_DIR / f"{method_name}_{split_name}_predictions.csv",
            index=False,
        )
    return rows


def main() -> None:
    args = parse_args()
    splits = load_splits(args.data_path, args.resplit)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vectorizer = CountVectorizer(
        lowercase=True,
        max_features=args.max_features,
        min_df=args.min_df,
        token_pattern=r"(?u)\b\w+\b",
    )
    train_counts = vectorizer.fit_transform(splits["train"]["text"]).tocsr()
    valid_counts = vectorizer.transform(splits["valid"]["text"]).tocsr()
    test_counts = vectorizer.transform(splits["test"]["text"]).tocsr()

    with (OUTPUT_DIR / "task1_vocabulary.json").open("w", encoding="utf-8") as file:
        json.dump(vectorizer.vocabulary_, file, ensure_ascii=True, sort_keys=True)

    results = []
    results.extend(
        run_experiment(
            "binary_bow",
            make_binary(train_counts),
            make_binary(valid_counts),
            make_binary(test_counts),
            splits,
            args.c,
        )
    )
    results.extend(
        run_experiment(
            "word_frequency",
            train_counts,
            valid_counts,
            test_counts,
            splits,
            args.c,
        )
    )

    results_dataframe = pd.DataFrame(results)
    results_dataframe.to_csv(OUTPUT_DIR / "task1_results.csv", index=False)

    print(
        f"Train/validation/test sizes: {len(splits['train'])}/"
        f"{len(splits['valid'])}/{len(splits['test'])}"
    )
    print(f"Vocabulary size: {train_counts.shape[1]}")
    print(results_dataframe.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
