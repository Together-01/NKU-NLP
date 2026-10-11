import argparse
import gc
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


RANDOM_STATE = 67
MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 64
LABELS = ("business", "politics", "sports")
DEFAULT_OUTPUT_DIR = Path("models/task3/bert_final")
DEFAULT_BEST_OUTPUT_DIR = Path("models/task3/bert_best")
DEFAULT_RESULTS_PATH = Path("outputs/task3_results.csv")
DEFAULT_HISTORY_PATH = Path("outputs/task3_training_history.csv")
DEFAULT_PREDICTIONS_PATH = Path("outputs/task3_bert_test_predictions.csv")
SPLIT_PATHS = {
    "train": Path("data/nyt_train.csv"),
    "valid": Path("data/nyt_valid.csv"),
    "test": Path("data/nyt_test.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT for NYT classification.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--best-output-dir",
        type=Path,
        default=DEFAULT_BEST_OUTPUT_DIR,
        help="Directory for the validation-best checkpoint.",
    )
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--history-path", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional local bert-base-uncased directory.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate an existing final checkpoint without training.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional training subset size for smoke tests.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional validation/test subset size for smoke tests.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate_args(args: argparse.Namespace) -> None:
    positive_arguments = {
        "epochs": args.epochs,
        "max_length": args.max_length,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "threads": args.threads,
    }
    for argument_name, value in positive_arguments.items():
        if value < 1:
            raise ValueError(f"--{argument_name.replace('_', '-')} must be at least 1")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    for argument_name in ("max_train_samples", "max_eval_samples"):
        value = getattr(args, argument_name)
        if value is not None and value < 1:
            raise ValueError(f"--{argument_name.replace('_', '-')} must be at least 1")
    if not args.eval_only and args.output_dir.resolve() == args.best_output_dir.resolve():
        raise ValueError("--output-dir and --best-output-dir must be different")


def find_local_model() -> Path | None:
    snapshot_root = (
        Path.home()
        / ".cache/huggingface/hub/models--bert-base-uncased/snapshots"
    )
    if not snapshot_root.exists():
        return None
    required_files = {"config.json", "vocab.txt", "pytorch_model.bin"}
    for snapshot in sorted(snapshot_root.iterdir(), reverse=True):
        if snapshot.is_dir() and all((snapshot / name).exists() for name in required_files):
            return snapshot
    return None


def resolve_model_source(model_path: Path | None) -> str | Path:
    if model_path is not None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        return model_path
    cached_model = find_local_model()
    return cached_model if cached_model is not None else MODEL_NAME


def load_splits(
    max_train_samples: int | None,
    max_eval_samples: int | None,
) -> dict[str, pd.DataFrame]:
    splits = {}
    for split_name, path in SPLIT_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run Task 1 data splitting first.")
        dataframe = pd.read_csv(path)
        missing_columns = {"text", "label"}.difference(dataframe.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path} is missing required columns: {missing}")
        dataframe = dataframe.loc[:, ["text", "label"]].dropna().copy()
        dataframe["text"] = dataframe["text"].astype(str)
        dataframe["label"] = dataframe["label"].astype(str)
        sample_limit = max_train_samples if split_name == "train" else max_eval_samples
        if sample_limit is not None:
            dataframe = dataframe.iloc[:sample_limit]
        splits[split_name] = dataframe.reset_index(drop=True)
    return splits


class EncodedTextDataset(Dataset):
    def __init__(
        self,
        texts: pd.Series,
        labels: pd.Series,
        tokenizer,
        label_to_id: dict[str, int],
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            texts.tolist(),
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = torch.tensor(
            [label_to_id[label] for label in labels],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {name: values[index] for name, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def make_data_loaders(
    splits: dict[str, pd.DataFrame],
    tokenizer,
    label_to_id: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, DataLoader]:
    datasets = {
        split_name: EncodedTextDataset(
            dataframe["text"],
            dataframe["label"],
            tokenizer,
            label_to_id,
            args.max_length,
        )
        for split_name, dataframe in splits.items()
    }
    generator = torch.Generator().manual_seed(RANDOM_STATE)
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.train_batch_size,
            shuffle=True,
            generator=generator,
        ),
        "valid": DataLoader(
            datasets["valid"],
            batch_size=args.eval_batch_size,
            shuffle=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.eval_batch_size,
            shuffle=False,
        ),
    }


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


def evaluate(
    model,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    labels = []
    total_loss = 0.0
    total_examples = 0
    with torch.inference_mode():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            batch_size = batch["labels"].size(0)
            total_loss += outputs.loss.item() * batch_size
            total_examples += batch_size
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())

    prediction_array = np.asarray(predictions)
    label_array = np.asarray(labels)
    metrics = {
        "loss": total_loss / total_examples,
        "accuracy": accuracy_score(label_array, prediction_array),
        "macro_f1": f1_score(label_array, prediction_array, average="macro"),
    }
    return metrics, prediction_array, label_array


def train_one_epoch(
    model,
    data_loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    gradient_accumulation_steps: int,
    epoch: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    epoch_start = time.monotonic()

    for batch_index, batch in enumerate(data_loader, start=1):
        batch = move_batch_to_device(batch, device)
        outputs = model(**batch)
        total_loss += outputs.loss.item()
        (outputs.loss / gradient_accumulation_steps).backward()

        should_step = (
            batch_index % gradient_accumulation_steps == 0
            or batch_index == len(data_loader)
        )
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if batch_index % 100 == 0 or batch_index == len(data_loader):
            elapsed = time.monotonic() - epoch_start
            print(
                f"Epoch {epoch}: batch {batch_index}/{len(data_loader)}, "
                f"loss={total_loss / batch_index:.4f}, elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    return total_loss / len(data_loader)


def save_training_config(
    output_dir: Path,
    args: argparse.Namespace,
    label_to_id: dict[str, int],
    model_source: str | Path,
    splits: dict[str, pd.DataFrame],
    checkpoint_role: str,
) -> None:
    config = {
        "pretrained_model": MODEL_NAME,
        "model_source": str(model_source),
        "epochs": args.epochs,
        "max_length": args.max_length,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "random_state": RANDOM_STATE,
        "label_to_id": label_to_id,
        "split_sizes": {name: len(dataframe) for name, dataframe in splits.items()},
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "checkpoint_role": checkpoint_role,
    }
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=True)


def validate_eval_checkpoint(
    output_dir: Path,
    args: argparse.Namespace,
    splits: dict[str, pd.DataFrame],
) -> None:
    config_path = output_dir / "training_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Checkpoint training config not found: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    expected_split_sizes = {name: len(dataframe) for name, dataframe in splits.items()}
    mismatches = []
    for key, expected in (
        ("pretrained_model", MODEL_NAME),
        ("epochs", args.epochs),
        ("max_length", args.max_length),
        ("split_sizes", expected_split_sizes),
        ("checkpoint_role", "final"),
    ):
        if config.get(key) != expected:
            mismatches.append(f"{key}={config.get(key)!r} (expected {expected!r})")
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"Checkpoint is incompatible with this evaluation: {details}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(RANDOM_STATE)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; PyTorch threads: {torch.get_num_threads()}", flush=True)

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.history_path.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    splits = load_splits(args.max_train_samples, args.max_eval_samples)
    observed_labels = set().union(*(set(split["label"]) for split in splits.values()))
    unexpected_labels = observed_labels.difference(LABELS)
    if unexpected_labels:
        unexpected = ", ".join(sorted(unexpected_labels))
        raise ValueError(f"Unexpected NYT labels: {unexpected}")
    labels = list(LABELS)
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    if args.eval_only:
        if not args.output_dir.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {args.output_dir}")
        validate_eval_checkpoint(args.output_dir, args, splits)
        model_source: str | Path = args.output_dir
    else:
        model_source = resolve_model_source(args.model_path)

    local_only = isinstance(model_source, Path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        local_files_only=local_only,
    )
    data_loaders = make_data_loaders(splits, tokenizer, label_to_id, args)

    if not args.eval_only:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=len(labels),
            id2label=id_to_label,
            label2id=label_to_id,
            local_files_only=local_only,
        ).to(device)
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        updates_per_epoch = math.ceil(
            len(data_loaders["train"]) / args.gradient_accumulation_steps
        )
        total_updates = updates_per_epoch * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_updates * args.warmup_ratio),
            num_training_steps=total_updates,
        )

        best_macro_f1 = -1.0
        history = []
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                data_loaders["train"],
                optimizer,
                scheduler,
                device,
                args.gradient_accumulation_steps,
                epoch,
            )
            validation_metrics, _, _ = evaluate(model, data_loaders["valid"], device)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_metrics["loss"],
                    "validation_accuracy": validation_metrics["accuracy"],
                    "validation_macro_f1": validation_metrics["macro_f1"],
                }
            )
            pd.DataFrame(history).to_csv(args.history_path, index=False)
            print(
                f"Epoch {epoch} validation: accuracy={validation_metrics['accuracy']:.4f}, "
                f"macro_f1={validation_metrics['macro_f1']:.4f}",
                flush=True,
            )
            if validation_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = validation_metrics["macro_f1"]
                args.best_output_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(args.best_output_dir)
                tokenizer.save_pretrained(args.best_output_dir)
                save_training_config(
                    args.best_output_dir,
                    args,
                    label_to_id,
                    model_source,
                    splits,
                    "validation_best",
                )
                print(f"Saved best model from epoch {epoch}", flush=True)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        save_training_config(
            args.output_dir,
            args,
            label_to_id,
            model_source,
            splits,
            "final",
        )
        del model, optimizer, scheduler
        gc.collect()
        final_validation = history[-1]
    else:
        history = (
            pd.read_csv(args.history_path).to_dict("records")
            if args.history_path.exists()
            else []
        )
        if history:
            final_validation = history[-1]
        else:
            final_validation = {
                "validation_accuracy": float("nan"),
                "validation_macro_f1": float("nan"),
            }

    final_model = AutoModelForSequenceClassification.from_pretrained(
        args.output_dir,
        local_files_only=True,
    ).to(device)
    test_metrics, test_predictions, _ = evaluate(
        final_model,
        data_loaders["test"],
        device,
    )

    prediction_output = splits["test"].copy()
    prediction_output["label_id"] = prediction_output["label"].map(label_to_id)
    prediction_output["prediction_id"] = test_predictions
    prediction_output["prediction"] = [id_to_label[index] for index in test_predictions]
    prediction_output.to_csv(args.predictions_path, index=False)

    results = pd.DataFrame(
        [
            {
                "method": "bert_base_uncased",
                "split": "validation",
                "accuracy": final_validation["validation_accuracy"],
                "macro_f1": final_validation["validation_macro_f1"],
                "epoch": args.epochs,
                "max_length": args.max_length,
                "checkpoint": "final",
            },
            {
                "method": "bert_base_uncased",
                "split": "test",
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "epoch": args.epochs,
                "max_length": args.max_length,
                "checkpoint": "final",
            },
        ]
    )
    results.to_csv(args.results_path, index=False)
    print(results.to_string(index=False, float_format="%.4f"), flush=True)


if __name__ == "__main__":
    main()
