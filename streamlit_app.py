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
MODEL_LABELS = {
    "linear_svm": "線性 SVM",
    "logistic_regression": "邏輯斯迴歸",
}
CTA_MESSAGE = "即刻體驗模型、比較指標並檢視資料洞察。"
SAMPLE_MESSAGES = {
    "抽獎詐騙": "Congratulations! You've won a free ticket. Call now to claim.",
    "帳單通知": "Reminder: Your invoice will be charged tomorrow unless you cancel.",
    "惡意連結": "Claim urgent refund at http://scam.link within 1 hour to avoid penalty.",
    "日常聊天": "Hey, are we still meeting for coffee this afternoon?",
}


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


@st.cache_data
def load_dataset_overview() -> Dict[str, Any]:
    if not RAW_DATA_PATH.exists():
        return {}
    df = load_dataset()
    label_counts = df["label"].value_counts().rename({1: "spam", 0: "ham"})
    avg_length = df["message"].str.len().mean()
    return {
        "count": len(df),
        "label_counts": label_counts.to_dict(),
        "avg_length": avg_length,
    }


def render_hero(metadata: Dict[str, Any]) -> None:
    st.markdown(
        """
        <div style="
            background: linear-gradient(120deg,#2b5876,#4e4376);
            padding: 2.5rem 2.75rem;
            border-radius: 20px;
            color: #ffffff;
        ">
            <h1 style="margin-bottom: 0.5rem;">📬 垃圾郵件分類控制台</h1>
            <p style="font-size: 1.05rem; margin-bottom: 0.75rem;">
                使用 TF-IDF 特徵工程與雙模型比較（線性 SVM、邏輯斯迴歸），快速辨識垃圾簡訊。
            </p>
            <div style="display: flex; gap: 1.5rem; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 0.9rem; opacity: 0.8;">訓練樣本數</span>
                    <h3 style="margin: 0;">{train}</h3>
                </div>
                <div>
                    <span style="font-size: 0.9rem; opacity: 0.8;">驗證樣本數</span>
                    <h3 style="margin: 0;">{test}</h3>
                </div>
                <div>
                    <span style="font-size: 0.9rem; opacity: 0.8;">快速導覽</span>
                    <h3 style="margin: 0;">{cta}</h3>
                </div>
            </div>
        </div>
        """.format(
            train=metadata.get("train_size", "N/A"),
            test=metadata.get("test_size", "N/A"),
            cta=CTA_MESSAGE,
        ),
        unsafe_allow_html=True,
    )


def render_prediction_tab(model_key: str, metrics: Dict[str, Any]) -> None:
    pipeline = load_pipeline(model_key)
    classifier = pipeline.named_steps["classifier"]
    vectorizer = pipeline.named_steps["vectorizer"]

    st.subheader("體驗即時分類")
    st.caption("輸入郵件內容或套用常見範例，模型會即時回報結果與信心分數。")

    col_msg, col_result = st.columns([2, 1])
    with col_msg:
        with st.expander("插入範例訊息", expanded=False):
            sample_choice = st.radio(
                "選擇範例（可自行編輯後再送出）",
                options=list(SAMPLE_MESSAGES.values()),
                format_func=lambda text: next(
                    label for label, content in SAMPLE_MESSAGES.items() if content == text
                ),
                index=0,
            )
        user_input = st.text_area(
            "郵件內容",
            sample_choice,
            height=180,
        )

    with col_result:
        st.markdown("#### 模型選擇")
        st.markdown(f"**{MODEL_LABELS[model_key]}**")
        st.markdown("---")
        run_prediction = st.button("🚀 執行分類", type="primary")

        if run_prediction:
            features = vectorizer.transform([user_input])
            prediction = classifier.predict(features)[0]
            label = "📛 Spam" if prediction == 1 else "✅ Ham"
            st.markdown(f"### {label}")

            if hasattr(classifier, "predict_proba"):
                proba = classifier.predict_proba(features)[0]
                st.metric("Spam 機率", f"{proba[1]:.2%}", delta=None)
                st.metric("Ham 機率", f"{proba[0]:.2%}", delta=None)
            elif hasattr(classifier, "decision_function"):
                score = classifier.decision_function(features)[0]
                st.metric("Decision Function 分數", f"{score:.4f}")
            else:
                st.info("此模型無機率輸出，僅供二元分類判斷。")
        else:
            st.info("按下「🚀 執行分類」即可查看結果。")


def render_metrics_tab(metrics: Dict[str, Any]) -> None:
    st.subheader("模型表現總覽")
    metric_rows = []
    for key, payload in metrics.items():
        if key == "metadata":
            continue
        metric_rows.append(
            {
                "模型": MODEL_LABELS.get(key, key),
                "Accuracy": payload["accuracy"],
                "Precision": payload["precision"],
                "Recall": payload["recall"],
                "F1": payload["f1"],
            }
        )

    metric_df = pd.DataFrame(metric_rows).set_index("模型")

    col_cards = st.columns(len(metric_df))
    for col, (model_name, row) in zip(col_cards, metric_df.iterrows()):
        with col:
            st.metric("模型", model_name)
            st.metric("Accuracy", f"{row['Accuracy']:.3f}")
            st.metric("F1", f"{row['F1']:.3f}")
            st.caption(f"Precision: {row['Precision']:.3f} · Recall: {row['Recall']:.3f}")

    st.markdown("### 指標趨勢比較")
    st.dataframe(metric_df.style.format("{:.3f}"), use_container_width=True)

    st.markdown("### 詳細分類報告")
    tabs = st.tabs(list(metric_df.index))
    for tab, key in zip(tabs, [k for k in metrics.keys() if k != "metadata"]):
        with tab:
            st.code(metrics[key]["classification_report"], language="text")


def render_dataset_tab() -> None:
    st.subheader("資料探索")
    overview = load_dataset_overview()
    if not overview:
        st.info("尚未下載資料集，請先執行訓練腳本。")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("樣本總數", f"{overview['count']}")
    spam_count = overview["label_counts"].get("spam", 0)
    ham_count = overview["label_counts"].get("ham", 0)
    col2.metric("Spam / Ham", f"{spam_count} / {ham_count}")
    col3.metric("平均訊息長度", f"{overview['avg_length']:.1f} 字元")

    st.markdown("### 隨機樣本")
    st.caption("保護個資：資料來源為公開 SMS 垃圾訊息資料集。")
    sample_df = load_sample_data(limit=10)
    st.table(sample_df)


def render_project_tab() -> None:
    st.subheader("專案使用指南")
    st.markdown(
        """
        - ✅ **準備環境**：`python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
        - 🛠️ **重新訓練模型**：`python -m src.train`（含資料下載與報表產生）
        - 🌐 **啟動介面**：`streamlit run streamlit_app.py`
        - 🚀 **部署建議**：使用 Streamlit Community Cloud，並記得將部署網址補回 README。
        - 📦 **專案結構**：`src/` 為資料與模型流程、`reports/` 存放指標、`artifacts/` 儲存模型。
        """
    )
    st.info("提示：若首次啟動，請先執行訓練腳本以產生模型與報告。")


def main() -> None:
    st.set_page_config(page_title="垃圾郵件分類控制台", page_icon="📬", layout="wide")

    metrics = load_metrics()
    metadata = metrics.get("metadata", {})
    render_hero(metadata)

    model_key = st.selectbox(
        "想要比較的模型",
        options=list(MODEL_LABELS.keys()),
        format_func=lambda key: MODEL_LABELS.get(key, key),
        index=0,
        help="可切換不同模型以查看指標與預測結果。",
    )

    tab_predict, tab_metrics, tab_dataset, tab_project = st.tabs(
        ["即時預測", "模型洞察", "資料探索", "專案指南"]
    )

    with tab_predict:
        render_prediction_tab(model_key, metrics)

    with tab_metrics:
        render_metrics_tab(metrics)

    with tab_dataset:
        render_dataset_tab()

    with tab_project:
        render_project_tab()

    st.markdown("---")
    st.caption(
        "資料來源：Hands-On Artificial Intelligence for Cybersecurity - SMS Spam Dataset."
    )


if __name__ == "__main__":
    main()
