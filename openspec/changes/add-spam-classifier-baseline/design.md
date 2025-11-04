# 垃圾郵件分類設計草案（合併階段）

## 資料流程
- 來源：`sms_spam_no_header.csv`（CSV，兩欄：label 與 message）。
- 下載策略：首次執行時透過 HTTP 抓取並儲存於專案內部的資料目錄（例如 `data/raw/`）；再次執行時優先使用快取。
- 前處理：去除缺值、統一大小寫、移除非必要符號，將標籤欄位標準化為二元（spam / ham）。
- 切分：使用固定隨機種子進行訓練/驗證分割（例如 80/20），確保可重複實驗。

## 特徵工程
- 文字向量化：採用 TF-IDF（`scikit-learn` 的 `TfidfVectorizer`），限制最小/最大字詞頻率以減少雜訊。
- 可能的前處理：停用詞移除、Stemming 或 Lemmatization（視需求）；在 Phase Baseline 記錄採用的具體策略。

## 模型策略
- 模型管線：共用資料前處理與 TF-IDF 特徵向量，建立可重複執行的訓練流程。
- 線性 SVM：採用 `LinearSVC` 作為基準模型，調整正規化參數以避免過度擬合。
- 邏輯斯迴歸：使用 `LogisticRegression`（liblinear 或 saga solver），與 SVM 共用向量化輸入。
- 評估指標：accuracy、precision、recall、F1-score；額外記錄混淆矩陣做錯誤分析。
- 結果比較：產出表格或圖表比較兩種模型的指標，並討論差異與可用性。

## 專案結構建議
- `data/`：原始與處理後資料。
- `notebooks/` 或 `experiments/`：實驗記錄（選擇性）。
- `src/`：資料載入、特徵工程、模型訓練與評估程式碼。
- `reports/`：模型評估報告（含指標摘要與後續建議）。

## 後續擴充構想
- 可加入其他演算法（如 Naive Bayes、深度學習）作比較。
- 部署考量（API、批次處理或即時服務）尚未定義。
- 監控與回訓：待確認產品需求後另行規劃。
