## ADDED Requirements

### Requirement: 建立垃圾郵件分類流程
- 系統 MUST 從 `sms_spam_no_header.csv` 取得資料並建立可重現的前處理、特徵工程與訓練流程。
- 流程 MUST 使用相同特徵工程訓練線性 SVM 與邏輯斯迴歸模型，並輸出 accuracy、precision、recall、F1 指標。
- 產出 MUST 包含訓練腳本或記錄檔，確保流程可以重複執行。

#### Scenario: 訓練並評估兩種模型
- Given 開發者取得並快取 `sms_spam_no_header.csv` 資料集
- And 依照定義的前處理與向量化流程完成資料轉換
- When 執行訓練程序以建立線性 SVM 與邏輯斯迴歸模型
- Then 會生成含有 accuracy、precision、recall、F1 指標的輸出或報告
- And 報告中同時包含兩種模型的結果與必要的重現資訊

### Requirement: 比較並記錄模型結果
- 報告 MUST 彙整線性 SVM 與邏輯斯迴歸的指標差異，指出主要優勢與限制。
- 結果 MUST 包含下一步建議或需要調整的方向，以利後續擴充。

#### Scenario: 審閱模型比較報告
- Given 審閱者檢視最終的報告或記錄檔
- When 查閱模型指標摘要
- Then 可同時看到線性 SVM 與邏輯斯迴歸的指標並列比較
- And 報告內包含後續改善建議或待探索議題
