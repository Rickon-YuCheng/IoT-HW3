import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


DATA_URL = (
    "https://raw.githubusercontent.com/PacktPublishing/"
    "Hands-On-Artificial-Intelligence-for-Cybersecurity/"
    "refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
)

RAW_DATA_PATH = Path("data/raw/sms_spam_no_header.csv")


def ensure_directory(path: Path) -> None:
    """Create parent directory for the given path if it doesn't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def download_dataset(url: str = DATA_URL, dest: Path = RAW_DATA_PATH, timeout: int = 30) -> Path:
    """
    Download the spam dataset if it is not already cached locally.

    Returns the path to the CSV file.
    """
    ensure_directory(dest)

    if dest.exists():
        logging.info("Dataset already cached at %s", dest)
        return dest

    logging.info("Downloading dataset from %s", url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    dest.write_bytes(response.content)
    logging.info("Downloaded dataset to %s", dest)
    return dest


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the SMS spam dataset into a DataFrame.
    The dataset has two columns without headers: label and message.
    """
    dataset_path = path or RAW_DATA_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Call download_dataset() before loading."
        )

    df = pd.read_csv(dataset_path, names=["label", "message"])
    # Normalize label values for consistency
    df["label"] = df["label"].str.lower().map({"spam": 1, "ham": 0})
    df = df.dropna(subset=["label", "message"])
    return df
