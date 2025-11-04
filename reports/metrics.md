# 垃圾郵件分類模型比較報告

- 訓練樣本數：4459
- 驗證樣本數：1115

## 指標總覽

| 模型 | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| linear_svm | 0.983 | 0.983 | 0.983 | 0.983 |
| logistic_regression | 0.978 | 0.978 | 0.978 | 0.978 |

## 個別模型詳細報告
### linear_svm

```text
precision    recall  f1-score   support

         ham       0.99      0.99      0.99       966
        spam       0.94      0.93      0.94       149

    accuracy                           0.98      1115
   macro avg       0.96      0.96      0.96      1115
weighted avg       0.98      0.98      0.98      1115
```

### logistic_regression

```text
precision    recall  f1-score   support

         ham       0.99      0.98      0.99       966
        spam       0.90      0.94      0.92       149

    accuracy                           0.98      1115
   macro avg       0.94      0.96      0.95      1115
weighted avg       0.98      0.98      0.98      1115
```

## 後續建議

- 目前 F1 分數最佳的模型為 **linear_svm**，建議優先使用該模型於展示與後續實驗。
- 持續觀察資料集中 spam 與 ham 的比例，一旦新增資料需重新訓練並確認指標。
- 可嘗試加入更進階的特徵（如文字長度、關鍵字）或不同演算法以提升效能。
