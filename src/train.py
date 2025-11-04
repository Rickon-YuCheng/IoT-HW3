import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .data_utils import download_dataset, load_dataset
from .text_utils import normalize_message


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ARTIFACT_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    metrics: Dict[str, float]
    classification_report: str


@dataclass
class TrainingSummary:
    results: Dict[str, ModelResult]
    train_size: int
    test_size: int


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=normalize_message,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )


def build_models(random_state: int) -> Dict[str, Pipeline]:
    vectorizer = build_vectorizer()
    svm_clf = Pipeline(
        [
            ("vectorizer", vectorizer),
            (
                "classifier",
                LinearSVC(class_weight="balanced", random_state=random_state),
            ),
        ]
    )

    logreg_clf = Pipeline(
        [
            ("vectorizer", build_vectorizer()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )

    return {
        "linear_svm": svm_clf,
        "logistic_regression": logreg_clf,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def train_and_evaluate(random_state: int = 42, test_size: float = 0.2) -> TrainingSummary:
    df = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"].tolist(),
        df["label"].to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    models = build_models(random_state)
    results: Dict[str, ModelResult] = {}

    for name, pipeline in models.items():
        logging.info("Training model: %s", name)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            target_names=["ham", "spam"],
            zero_division=0,
        )
        results[name] = ModelResult(
            name=name,
            pipeline=pipeline,
            metrics=metrics,
            classification_report=report,
        )
        logging.info("%s metrics: %s", name, metrics)

    return TrainingSummary(results=results, train_size=len(X_train), test_size=len(X_test))


def save_artifacts(results: Dict[str, ModelResult]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        path = ARTIFACT_DIR / f"{name}_pipeline.joblib"
        joblib.dump(result.pipeline, path)
        logging.info("Saved %s pipeline to %s", name, path)


def save_reports(results: Dict[str, ModelResult], train_size: int, test_size: int) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        name: {
            **result.metrics,
            "classification_report": result.classification_report,
        }
        for name, result in results.items()
    }
    metrics_payload["metadata"] = {"train_size": train_size, "test_size": test_size}

    metrics_path = REPORTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False))
    logging.info("Saved metrics to %s", metrics_path)

    markdown_lines = [
        "# 垃圾郵件分類模型比較報告",
        "",
        f"- 訓練樣本數：{train_size}",
        f"- 驗證樣本數：{test_size}",
        "",
        "## 指標總覽",
        "",
        "| 模型 | Accuracy | Precision | Recall | F1 |",
        "| --- | --- | --- | --- | --- |",
    ]

    best_model = None
    best_f1 = -1.0
    for name, result in results.items():
        metrics = result.metrics
        markdown_lines.append(
            f"| {name} | "
            f"{metrics['accuracy']:.3f} | {metrics['precision']:.3f} | "
            f"{metrics['recall']:.3f} | {metrics['f1']:.3f} |"
        )
        if metrics["f1"] > best_f1:
            best_model = name
            best_f1 = metrics["f1"]

    markdown_lines.extend(
        [
            "",
            "## 個別模型詳細報告",
        ]
    )

    for name, result in results.items():
        markdown_lines.extend(
            [
                f"### {name}",
                "",
                "```text",
                result.classification_report.strip(),
                "```",
                "",
            ]
        )

    suggestions = [
        "持續觀察資料集中 spam 與 ham 的比例，一旦新增資料需重新訓練並確認指標。",
        "可嘗試加入更進階的特徵（如文字長度、關鍵字）或不同演算法以提升效能。",
    ]
    if best_model:
        suggestions.insert(
            0,
            f"目前 F1 分數最佳的模型為 **{best_model}**，建議優先使用該模型於展示與後續實驗。",
        )

    markdown_lines.extend(
        [
            "## 後續建議",
            "",
            *[f"- {item}" for item in suggestions],
            "",
        ]
    )

    markdown_path = REPORTS_DIR / "metrics.md"
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    logging.info("Saved human-readable report to %s", markdown_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train spam classification models.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume dataset already cached and skip download step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_download:
        download_dataset()

    summary = train_and_evaluate(random_state=args.random_state, test_size=args.test_size)

    save_artifacts(summary.results)
    save_reports(summary.results, train_size=summary.train_size, test_size=summary.test_size)


if __name__ == "__main__":
    main()
