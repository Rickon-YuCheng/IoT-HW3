# 垃圾郵件分類專案

這個專案示範如何以 TF-IDF 特徵工程搭配 **線性 SVM** 與 **邏輯斯迴歸** 兩種模型建置垃圾簡訊分類器，並透過 Streamlit 提供互動式展示。

- 📦 GitHub Repo（目前專案）
- 🌐 Streamlit Demo：<https://iot-hw3-fwtvcqvyp7rayp8u49brbs.streamlit.app/>

> Demo 網址請在部署完成後更新為自己的 Streamlit App 連結。

## 專案特色
- 以公開資料集 [`sms_spam_no_header.csv`](https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv) 為基礎。
- 整合線性 SVM 與邏輯斯迴歸，輸出 Accuracy、Precision、Recall、F1 指標並產生報告。
- 產生 `artifacts/` 模型檔與 `reports/metrics.json`、`reports/metrics.md` 比較報告。
- Streamlit 介面可即時輸入郵件內容進行分類、瀏覽指標與樣本資料。

## 專案結構
```
.
├── artifacts/                     # 訓練後的模型管線（由腳本產生）
├── data/
│   └── raw/                       # 原始資料集（由腳本下載）
├── reports/
│   ├── metrics.json               # 機器可讀的評估指標（由腳本產生）
│   └── metrics.md                 # 人類可讀的比較報告
├── src/
│   ├── __init__.py
│   ├── data_utils.py              # 資料下載與載入工具
│   ├── text_utils.py              # 文字前處理工具
│   └── train.py                   # 主訓練腳本（同時訓練 SVM 與邏輯回歸）
├── streamlit_app.py               # Streamlit 主程式
├── requirements.txt               # Python 依賴套件
└── README.md
```

## 快速開始
1. 建立虛擬環境並安裝套件：
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. 訓練並產生模型、報告：
   ```bash
   python -m src.train
   ```
3. 啟動本地 Streamlit：
   ```bash
   streamlit run streamlit_app.py
   ```

## Streamlit 部署建議
1. 將 `requirements.txt` 與 `streamlit_app.py` 推送到 GitHub。
2. 在 [Streamlit Community Cloud](https://streamlit.io/cloud) 建立專案並指定本 repo。
3. 設定主程式為 `streamlit_app.py`，確保訓練腳本先執行並提交產生的模型檔（或在 app 中加入自動下載與訓練邏輯）。
4. 部署完成後，回到 README 更新 Streamlit 網址。

## 評估報告
`python -m src.train` 會在 `reports/metrics.md` 生成詳細報告，包含：
- 每個模型的 accuracy、precision、recall、F1 指標
- `classification_report` 詳細分類結果
- 後續改善建議

## 後續規劃想法
- 加入更多模型（如 Naive Bayes、深度學習）。
- 建立 API 或批次處理管線，以利系統整合。
- 加入資料監控與自動再訓練流程。

## 參考
- [Hands-On Artificial Intelligence for Cybersecurity](https://www.packtpub.com/) Chapter 03 資料集。
- 官方 Streamlit 文件：<https://docs.streamlit.io/>
- scikit-learn 文件：<https://scikit-learn.org/stable/>
