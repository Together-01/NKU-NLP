import argparse
import csv
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


RANDOM_STATE = 67
VECTOR_SIZE = 100
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models/task2")
EMBEDDING_DIR = Path("embeddings")
SPLIT_PATHS = {
    "train": DATA_DIR / "nyt_train.csv",
    "valid": DATA_DIR / "nyt_valid.csv",
    "test": DATA_DIR / "nyt_test.csv",
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all Task 2 embedding experiments.")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("glove", "ag_word2vec", "nyt_word2vec"),
        default=("glove", "ag_word2vec", "nyt_word2vec"),
        help="Experiments to run (default: all three).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs for the two Word2Vec models.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Word2Vec worker threads. Use 1 for deterministic training.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain Word2Vec models even when saved models exist.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for Logistic Regression.",
    )
    parser.add_argument(
        "--glove-path",
        type=Path,
        default=EMBEDDING_DIR / "glove.6B.100d.txt",
        help="Path to the Stanford glove.6B.100d.txt file.",
    )
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class CsvSentenceCorpus(Iterable[list[str]]):
    def __init__(self, path: Path, text_column: str = "text") -> None:
        self.path = path
        self.text_column = text_column

    def __iter__(self) -> Iterator[list[str]]:
        with self.path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or self.text_column not in reader.fieldnames:
                raise ValueError(f"{self.path} has no '{self.text_column}' column")
            for row in reader:
                tokens = tokenize(row[self.text_column])
                if tokens:
                    yield tokens


def load_nyt_splits() -> dict[str, pd.DataFrame]:
    splits = {}
    for split_name, path in SPLIT_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run 'python main.py --resplit' before Task 2."
            )
        dataframe = pd.read_csv(path)
        required_columns = {"text", "label"}
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path} is missing required columns: {missing}")
        dataframe = dataframe.loc[:, ["text", "label"]].dropna().copy()
        dataframe["text"] = dataframe["text"].astype(str)
        dataframe["label"] = dataframe["label"].astype(str)
        splits[split_name] = dataframe.reset_index(drop=True)
    return splits


def load_glove_vectors(glove_path: Path):
    from gensim.models import KeyedVectors

    if not glove_path.exists():
        raise FileNotFoundError(
            f"Missing {glove_path}. Download glove.6B.zip from Stanford and extract "
            "glove.6B.100d.txt into the embeddings directory."
        )
    vectors = KeyedVectors.load_word2vec_format(
        str(glove_path),
        binary=False,
        no_header=True,
    )
    if vectors.vector_size != VECTOR_SIZE:
        raise ValueError(f"Expected 100-dimensional GloVe, got {vectors.vector_size}")
    return vectors


def load_or_train_word2vec(
    corpus_path: Path,
    model_path: Path,
    epochs: int,
    workers: int,
    retrain: bool,
):
    from gensim.models import Word2Vec

    model = None
    metadata_path = model_path.with_suffix(model_path.suffix + ".json")
    expected_metadata = {
        "corpus_path": str(corpus_path),
        "vector_size": VECTOR_SIZE,
        "window": 5,
        "min_count": 2,
        "sg": 1,
        "epochs": epochs,
        "workers": workers,
        "seed": RANDOM_STATE,
    }
    if model_path.exists() and not retrain:
        try:
            with metadata_path.open(encoding="utf-8") as file:
                metadata = json.load(file)
            if metadata == expected_metadata:
                model = Word2Vec.load(str(model_path))
            else:
                print(f"Ignoring cached model with incompatible metadata: {model_path}")
        except FileNotFoundError:
            print(f"Ignoring cached model without metadata: {model_path}")
        except (EOFError, OSError, ValueError) as error:
            print(f"Ignoring invalid cached model {model_path}: {error}")

    if model is None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model = Word2Vec(
            sentences=CsvSentenceCorpus(corpus_path),
            vector_size=VECTOR_SIZE,
            window=5,
            min_count=2,
            workers=workers,
            sg=1,
            epochs=epochs,
            seed=RANDOM_STATE,
        )
        model.save(str(model_path))
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(expected_metadata, file, indent=2, ensure_ascii=True)

    if model.wv.vector_size != VECTOR_SIZE:
        raise ValueError(f"Expected 100-dimensional Word2Vec, got {model.wv.vector_size}")
    return model.wv


def make_document_vectors(
    texts: pd.Series,
    keyed_vectors,
) -> tuple[np.ndarray, float, int]:
    document_vectors = np.zeros((len(texts), VECTOR_SIZE), dtype=np.float32)
    total_tokens = 0
    matched_tokens = 0
    empty_documents = 0

    for row_index, text in enumerate(texts):
        tokens = tokenize(text)
        total_tokens += len(tokens)
        known_tokens = [token for token in tokens if token in keyed_vectors]
        matched_tokens += len(known_tokens)
        if known_tokens:
            document_vectors[row_index] = np.mean(
                keyed_vectors[known_tokens],
                axis=0,
                dtype=np.float32,
            )
        else:
            empty_documents += 1

    coverage = matched_tokens / total_tokens if total_tokens else 0.0
    return document_vectors, coverage, empty_documents


def evaluate_method(
    method_name: str,
    keyed_vectors,
    splits: dict[str, pd.DataFrame],
    regularization_c: float,
) -> list[dict[str, str | float | int]]:
    features = {}
    coverage = {}
    empty_documents = {}
    for split_name, dataframe in splits.items():
        split_features, split_coverage, split_empty_documents = make_document_vectors(
            dataframe["text"], keyed_vectors
        )
        if not np.isfinite(split_features).all():
            raise ValueError(f"{method_name} produced non-finite {split_name} vectors")
        features[split_name] = split_features
        coverage[split_name] = split_coverage
        empty_documents[split_name] = split_empty_documents

    model = LogisticRegression(
        C=regularization_c,
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    model.fit(features["train"], splits["train"]["label"])

    rows = []
    for split_name in ("valid", "test"):
        predictions = model.predict(features[split_name])
        labels = splits[split_name]["label"]
        rows.append(
            {
                "method": method_name,
                "split": "validation" if split_name == "valid" else "test",
                "accuracy": accuracy_score(labels, predictions),
                "macro_f1": f1_score(labels, predictions, average="macro"),
                "token_coverage": coverage[split_name],
                "empty_documents": empty_documents[split_name],
                "vector_size": features[split_name].shape[1],
            }
        )

        prediction_output = splits[split_name].copy()
        prediction_output["prediction"] = predictions
        prediction_output.to_csv(
            OUTPUT_DIR
            / f"task2_{method_name}_{'validation' if split_name == 'valid' else 'test'}_predictions.csv",
            index=False,
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    splits = load_nyt_splits()

    methods = {
        "glove": lambda: load_glove_vectors(args.glove_path),
        "ag_word2vec": lambda: load_or_train_word2vec(
            DATA_DIR / "ag.csv",
            MODEL_DIR / "ag_news_word2vec.model",
            args.epochs,
            args.workers,
            args.retrain,
        ),
        "nyt_word2vec": lambda: load_or_train_word2vec(
            SPLIT_PATHS["train"],
            MODEL_DIR / "nyt_train_word2vec.model",
            args.epochs,
            args.workers,
            args.retrain,
        ),
    }

    all_results = []
    for method_name in args.methods:
        print(f"Running {method_name}...")
        keyed_vectors = methods[method_name]()
        method_results = evaluate_method(
            method_name,
            keyed_vectors,
            splits,
            args.c,
        )
        all_results.extend(method_results)
        print(pd.DataFrame(method_results).to_string(index=False, float_format="%.4f"))

    results_path = OUTPUT_DIR / "task2_results.csv"
    new_results = pd.DataFrame(all_results)
    if results_path.exists():
        existing_results = pd.read_csv(results_path)
        existing_results = existing_results[
            ~existing_results["method"].isin(new_results["method"])
        ]
        new_results = pd.concat([existing_results, new_results], ignore_index=True)
    method_order = {name: index for index, name in enumerate(methods)}
    split_order = {"validation": 0, "test": 1}
    new_results["_method_order"] = new_results["method"].map(method_order)
    new_results["_split_order"] = new_results["split"].map(split_order)
    new_results = new_results.sort_values(["_method_order", "_split_order"]).drop(
        columns=["_method_order", "_split_order"]
    )
    new_results.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
