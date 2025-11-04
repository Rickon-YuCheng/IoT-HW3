from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import streamlit as st

from src.data_utils import RAW_DATA_PATH, load_dataset


ARTIFACT_DIR = Path("artifacts")
REPORTS_PATH = Path("reports/metrics.json")


@st.cache_resource
def load_pipeline(model_key: str):
    path = ARTIFACT_DIR / f"{model_key}_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到模型檔案 {path}，請先執行 `python -m src.train` 產生模型。"
        )
    return joblib.load(path)


@st.cache_data
def load_metrics() -> Dict[str, Any]:
    if not REPORTS_PATH.exists():
        raise FileNotFoundError(
            f"找不到報告檔案 {REPORTS_PATH}，請先執行 `python -m src.train` 產生報表。"
        )
    return json.loads(REPORTS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_sample_data(limit: int = 5) -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        return pd.DataFrame(columns=["label", "message"])
    df = load_dataset()
    df["label"] = df["label"].map({1: "spam", 0: "ham"})
    return df.sample(n=min(limit, len(df)), random_state=42)


def main() -> None:
    st.set_page_config(page_title="垃圾郵件分類器", page_icon="📬", layout="wide")
    st.title("📬 垃圾郵件分類器")
    st.caption("使用 TF-IDF + 線性 SVM 與邏輯斯迴歸模型進行比較")

    metrics = load_metrics()
    metadata = metrics.get("metadata", {})
    st.sidebar.header("模型資訊")
    st.sidebar.write(f"訓練樣本數：{metadata.get('train_size', 'N/A')}")
    st.sidebar.write(f"驗證樣本數：{metadata.get('test_size', 'N/A')}")

    model_key = st.sidebar.selectbox(
        "選擇模型",
        options=["linear_svm", "logistic_regression"],
        format_func=lambda key: "線性 SVM" if key == "linear_svm" else "邏輯斯迴歸",
    )

    pipeline = load_pipeline(model_key)
    classifier = pipeline.named_steps["classifier"]

    st.header("即時預測")
    default_text = "Congratulations! You've won a free ticket. Call now to claim."
    user_input = st.text_area("輸入電子郵件內容", default_text, height=160)

    if st.button("進行分類", type="primary"):
        vectorizer = pipeline.named_steps["vectorizer"]
        features = vectorizer.transform([user_input])
        prediction = classifier.predict(features)[0]
        label = "Spam" if prediction == 1 else "Ham"

        st.subheader("預測結果")
        st.write(f"模型判定：**{label}**")

        if hasattr(classifier, "predict_proba"):
            proba = classifier.predict_proba(features)[0]
            st.write(
                f"Spam 機率：{proba[1]:.2%} · Ham 機率：{proba[0]:.2%}"
            )
        elif hasattr(classifier, "decision_function"):
            score = classifier.decision_function(features)[0]
            st.write(f"Decision function 分數：{score:.4f}")

    st.header("模型指標")
    metric_rows = []
    for key, payload in metrics.items():
        if key in ("metadata",):
            continue
        metric_rows.append(
            {
                "模型": "線性 SVM" if key == "linear_svm" else "邏輯斯迴歸",
                "Accuracy": payload["accuracy"],
                "Precision": payload["precision"],
                "Recall": payload["recall"],
                "F1": payload["f1"],
            }
        )

    metric_df = pd.DataFrame(metric_rows).set_index("模型")
    st.dataframe(metric_df.style.format("{:.3f}"), use_container_width=True)

    st.header("分類報告")
    report_text = metrics[model_key]["classification_report"]
    st.code(report_text, language="text")

    st.header("資料集範例")
    sample_df = load_sample_data(limit=5)
    if sample_df.empty:
        st.info("尚未下載資料集，請先執行訓練腳本。")
    else:
        st.table(sample_df)

    st.header("使用說明")
    st.markdown(
        """
        1. 先執行 `python -m src.train` 下載資料、訓練模型並生成報表。
        2. 使用 `streamlit run streamlit_app.py` 啟動本介面。
        3. 在 GitHub README 中放上 Streamlit 部署網址，供使用者體驗。
        """
    )


if __name__ == "__main__":
    main()
