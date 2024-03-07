from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# 函數: 評估分類模型性能
def classification_metrics(model, feature_test, label_test, model_name):
    """
    參數: 
    - model: 訓練完成的模型 (模型物件)
    - feature_test: 測試數據集的特徵 (numpy.ndarray 或 pandas.DataFrame)
    - label_test: 測試數據集對應的標籤 (numpy.ndarray 或 pandas.Series)
    - model_name: 模型名稱 (str)

    返回: 
    - 包含各種分類性能指標的字典 (dict)
      如果發生錯誤，則返回 None
    """
    
    try:
        # 在測試數據上進行預測
        predictions = model.predict(feature_test)

        # 計算評估模型性能的指標
        accuracy = accuracy_score(label_test, predictions)  # 準確度
        mcc = matthews_corrcoef(label_test, predictions)    # 馬修斯相關系數
        f1 = f1_score(label_test, predictions)              # F1 分數

        # 計算混淆矩陣
        confusion = confusion_matrix(label_test, predictions)
        sensitivity = confusion[1, 1] / (confusion[1, 1] + confusion[1, 0])  # 敏感度
        specificity = confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])  # 特異性

        # 創建字典來存儲模型性能指標結果
        performance_metrics = {
            'Accuracy': accuracy,
            'MCC': mcc,
            'F1 Score': f1,
            'Sensitivity': sensitivity,
            'Specificity': specificity
        }

        # 打印性能指標
        print(f"Metrics for {model_name}:")
        print("=" * 50)
        for key, value in performance_metrics.items():
            print(f"{key}: {value:.4f}")
        print("=" * 50 + "\n")

        return performance_metrics
    
    except Exception as e:
        # 錯誤處理
        print(f"An error occurred during classification metrics evaluation:\n{e}")
        return None

# 函數: 評估回歸模型性能
def regression_metrics(model, feature_test, target_test, model_name):
    """
    參數: 
    - model: 訓練完成的模型 (模型物件)
    - feature_test: 測試數據集的特徵 (numpy.ndarray 或 pandas.DataFrame) 
    - target_test: 測試數據集對應的目標值 (numpy.ndarray 或 pandas.Series)
    - model_name: 模型的名稱 (str)

    返回: 
    - 包含各種回歸性能指標的字典 (dict)
      如果發生錯誤，則返回 None
    """

    try:
        # 在測試數據上進行預測
        predictions = model.predict(feature_test)

        # 計算評估回歸模型性能的指標
        mae = mean_absolute_error(target_test, predictions)                 # 平均絕對誤差
        mse = mean_squared_error(target_test, predictions)                  # 均方誤差
        rmse = mean_squared_error(target_test, predictions, squared=False)  # 均方根誤差
        r2 = r2_score(target_test, predictions)                             # 決定係數
        pcc, p_value = pearsonr(target_test, predictions)                   # 計算皮爾森相關係數

        # 創建字典來存儲模型性能指標結果
        performance_metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R^2 Score': r2,
            'PCC': pcc
        }

        # 打印性能指標
        print(f"Metrics for {model_name}:")
        print("=" * 50)
        for key, value in performance_metrics.items():
            print(f"{key}: {value:.4f}")
        if p_value < 0.05:
            print(f"PCC p-value: {p_value:.4f} (significant)")
        else:
            print(f"PCC p-value: {p_value:.4f} (not significant)")
        print("=" * 50 + "\n")

        return performance_metrics
    
    except Exception as e:
        # 錯誤處理
        print(f"An error occurred during regression metrics evaluation:\n{e}")
        return None
