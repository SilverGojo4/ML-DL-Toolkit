from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd

# 函數: 使用隨機搜索的 XGBoost 分類模型
def xgb_classify_random_search(feature_train, label_train, param_distributions, 
                               model_random_state=42, search_random_state=42, 
                               scale_pos_weight=1, n_iter=200, cv=5, n_jobs=-1, verbose=2):
    """
    參數: 
    - feature_train: 訓練數據集的特徵 (numpy.ndarray 或 pandas.DataFrame) 
    - label_train: 訓練數據集對應的標籤 (numpy.ndarray 或 pandas.Series)
    - param_distributions: 隨機搜索中超參數的分佈 (dict)
    - model_random_state: 初始化 XGBClassifier 的隨機種子 (int, 預設為 42)
    - search_random_state: 初始化 RandomizedSearchCV 的隨機種子 (int, 預設為 42)
    - scale_pos_weight: 正樣本的權重 (float, 預設為 1)，用於不平衡數據集，常見的設置值是使用 (負樣本數量)/(正樣本數量)
    - n_iter: 抽樣的參數設置數量 (int, 預設為 200)
    - cv: 交叉驗證的折數 (int, 預設為 5)
    - n_jobs: 並行運行的作業數 (int, 預設為 -1, 使用所有處理器)
    - verbose: 日誌詳細程度 (int, 預設為 2)
    
    返回: 
    - 通過 RandomizedSearchCV 找到的最佳 XGBoost 分類器 (XGBClassifier 物件)
      如果訓練過程中發生錯誤，則返回 None
    """

    # 驗證 feature_train 和 label_train 的長度是否匹配
    if len(feature_train) != len(label_train):
        raise ValueError("The lengths of feature_train and label_train do not match")

    # 驗證 param_distributions 是否為 dict 且不為空
    if not isinstance(param_distributions, dict) or not param_distributions:
        raise ValueError("param_distributions must be a non-empty dictionary")

    # 驗證 n_iter、cv、和 n_jobs 是否為正整數
    if not all(isinstance(param, int) and param > 0 for param in [n_iter, cv]) or not isinstance(n_jobs, int):
        raise ValueError("n_iter and cv must be positive integers, n_jobs must be an integer")

    try:
        # 初始化 XGBClassifier
        xgb = XGBClassifier(random_state=model_random_state, 
                            scale_pos_weight=scale_pos_weight, 
                            use_label_encoder=False, eval_metric='logloss')

        # 創建 RandomizedSearchCV 對象
        xgb_random_search = RandomizedSearchCV(estimator=xgb, 
                                               param_distributions=param_distributions, 
                                               random_state=search_random_state, 
                                               n_iter=n_iter, cv=cv, 
                                               n_jobs=n_jobs, verbose=verbose)

        # 對訓練數據進行擬合
        xgb_random_search.fit(feature_train, label_train)

        # 獲得最佳估計器 (最佳 XGBoost 模型)
        best_xgb_model = xgb_random_search.best_estimator_

        # 打印最佳 XGBoost 分類模型的詳細資訊
        print("Best XGBoost Classifier Model")
        print("=" * 80)
        print("Best parameters:")
        for param, value in xgb_random_search.best_params_.items():
            print(f"{param}: {value}")
        print("-" * 80)
        print(f"Validation Set Score: {xgb_random_search.best_score_}")
        print("=" * 80 + "\n")

        return best_xgb_model

    except Exception as e:
        # 錯誤處理
        print(f"An error occurred during the XGBoost Classification training process:\n{e}")
        return None

# 函數: 使用隨機搜索的 XGBoost 回歸模型
def xgb_regress_random_search(feature_train, target_train, param_distributions, 
                              model_random_state=42, search_random_state=42, 
                              n_iter=200, cv=5, n_jobs=-1, verbose=2):
    """
    參數: 
    - feature_train: 訓練數據集的特徵 (numpy.ndarray 或 pandas.DataFrame) 
    - target_train: 訓練數據集對應的目標值 (numpy.ndarray 或 pandas.Series)
    - param_distributions: 隨機搜索中超參數的分佈 (dict)
    - model_random_state: 初始化 XGBRegressor 的隨機種子 (int, 預設為 42)
    - search_random_state: 初始化 RandomizedSearchCV 的隨機種子 (int, 預設為 42)
    - n_iter: 抽樣的參數設置數量 (int, 預設為 200)
    - cv: 交叉驗證的折數 (int, 預設為 5)
    - n_jobs: 並行運行的作業數 (int, 預設為 -1, 使用所有處理器)
    - verbose: 日誌詳細程度 (int, 預設為 2)
    
    返回: 
    - 通過 RandomizedSearchCV 找到的最佳 XGBoost 回歸器 (XGBRegressor 物件)
      如果訓練過程中發生錯誤，則返回 None
    """

    # 驗證 feature_train 和 label_train 的長度是否匹配
    if len(feature_train) != len(target_train):
        raise ValueError("The lengths of feature_train and target_train do not match")

    # 驗證 param_distributions 是否為 dict 且不為空
    if not isinstance(param_distributions, dict) or not param_distributions:
        raise ValueError("param_distributions must be a non-empty dictionary")

    # 驗證 n_iter、cv、和 n_jobs 是否為正整數
    if not all(isinstance(param, int) and param > 0 for param in [n_iter, cv]) or not isinstance(n_jobs, int):
        raise ValueError("n_iter and cv must be positive integers, n_jobs must be an integer")

    try:
        # 初始化 XGBRegressor
        xgb = XGBRegressor(random_state=model_random_state)

        # 創建 RandomizedSearchCV 對象
        xgb_random_search = RandomizedSearchCV(estimator=xgb, 
                                               param_distributions=param_distributions, 
                                               random_state=search_random_state, 
                                               n_iter=n_iter, cv=cv, 
                                               n_jobs=n_jobs, verbose=verbose)

        # 對訓練數據進行擬合
        xgb_random_search.fit(feature_train, target_train)

        # 獲得最佳估計器 (最佳 XGBoost 模型)
        best_xgb_model = xgb_random_search.best_estimator_

        # 打印最佳 XGBoost 回歸模型的詳細資訊
        print("Best XGBoost Regressor Model")
        print("=" * 80)
        print("Best parameters:")
        for param, value in xgb_random_search.best_params_.items():
            print(f"{param}: {value}")
        print("-" * 80)
        print(f"Validation Set Score: {xgb_random_search.best_score_}")
        print("=" * 80 + "\n")

        return best_xgb_model
    
    except Exception as e:
        # 錯誤處理
        print(f"An error occurred during the XGBoost Regression training process:\n{e}")
        return None

# 函數: 輸出特徵重要性至 CSV 檔案
def xgb_importance_score_to_csv(model, feature_names, output_file):
    """
    參數: 
    - model: 訓練完成的 XGBoost 模型 (XGBClassifier 或 XGBRegressor 物件)
    - feature_names: 特徵名稱列表 (list)
    - output_file: 輸出 CSV 檔案的路徑 (str)

    返回: 
    - 排序後的特徵重要性 DataFrame (pandas.DataFrame)
      如果發生錯誤，則返回 None
    """
    
    try:
        # 從模型中獲取特徵重要性
        importances = model.feature_importances_

        # 計算包括零值的平均重要性分數
        average_importance_incl_zero = np.mean(importances)

        # 計算不包括零值的平均重要性分數
        non_zero_importances = [imp for imp in importances if imp > 0]
        average_importance_excl_zero = np.mean(non_zero_importances) if non_zero_importances else 0

        # 中位數重要性分數
        median_importance = np.median(importances)

        # 計算超過平均重要性分數的特徵數量
        features_above_avg_incl_zero = sum(importances > average_importance_incl_zero)
        features_above_avg_excl_zero = sum(importances > average_importance_excl_zero)

        # 計算超過中位數重要性分數的特徵數量
        features_above_median = sum(importances > median_importance)

        print("Best XGBoost Model")
        print("=" * 80)
        print("Importance score (including zero):")
        print(f"Average importance score: {average_importance_incl_zero}")
        print(f"Number of features above average importance score: {features_above_avg_incl_zero}")
        print("-" * 80)
        print("Importance score (excluding zero):")
        print(f"Average importance score: {average_importance_excl_zero}")
        print(f"Number of features above average importance score: {features_above_avg_excl_zero}")
        print("-" * 80)
        print("Importance score (Median):")
        print(f"Median importance score: {median_importance}")
        print(f"Number of features above median importance score: {features_above_median}")
        print("=" * 80 + "\n")

        # 創建 DataFrame 來存儲特徵重要性
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

        # 按照重要性分數降序對 DataFrame 進行排序
        sorted_feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # 添加重要性排名列
        sorted_feature_importance_df['Importance rank'] = range(1, len(sorted_feature_importance_df) + 1)

        # 將排序後的 DataFrame 保存到 CSV 檔案
        sorted_feature_importance_df.to_csv(output_file, index=False)

        return sorted_feature_importance_df

    except Exception as e:
        # 錯誤處理
        print(f"An error occurred during exporting feature importance scores to CSV:\n{e}")
        return None
    