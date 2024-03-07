from sklearn.svm import SVC, SVR
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd

# 函數: 使用隨機搜索的 SVM 分類模型
def svm_classify_random_search(feature_train, label_train, param_distributions, 
                               model_random_state=42, search_random_state=42, 
                               class_weight=None, n_iter=200, cv=5, n_jobs=-1, verbose=2):
    """
    參數: 
    - feature_train: 訓練數據集的特徵 (numpy.ndarray 或 pandas.DataFrame) 
    - label_train: 訓練數據集對應的標籤 (numpy.ndarray 或 pandas.Series)
    - param_distributions: 隨機搜索中超參數的分佈 (dict)
    - model_random_state: 初始化 SVC 的隨機種子 (int, 預設為 42)
    - search_random_state: 初始化 RandomizedSearchCV 的隨機種子 (int, 預設為 42)
    - class_weight: 類別權重參數 (dict 或 'balanced', 預設為 None)
    - n_iter: 抽樣的參數設置數量 (int, 預設為 200)
    - cv: 交叉驗證的折數 (int, 預設為 5)
    - n_jobs: 並行運行的作業數 (int, 預設為 -1, 使用所有處理器)
    - verbose: 日誌詳細程度 (int, 預設為 2)

    返回: 
    - 通過 RandomizedSearchCV 找到的最佳 SVM 分類器 (SVC 物件)
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
        # 初始化 SVC
        svm = SVC(random_state=model_random_state, 
                  class_weight=class_weight)

        # 創建 RandomizedSearchCV 對象
        svm_random_search = RandomizedSearchCV(estimator=svm, 
                                               param_distributions=param_distributions, 
                                               random_state=search_random_state, 
                                               n_iter=n_iter, cv=cv, 
                                               n_jobs=n_jobs, verbose=verbose)

        # 對訓練數據進行擬合
        svm_random_search.fit(feature_train, label_train)

        # 獲得最佳估計器 (最佳 SVM 分類模型)
        best_svm_classifier = svm_random_search.best_estimator_

        # 打印最佳 SVM 分類模型的詳細資訊
        print("Best SVM Classifier Model")
        print("=" * 80)
        print("Best parameters:")
        for param, value in svm_random_search.best_params_.items():
            print(f"{param}: {value}")
        print("-" * 80)
        print(f"Validation Set Score: {svm_random_search.best_score_}")
        print("=" * 80 + "\n")

        return best_svm_classifier
    
    except Exception as e:
        # 錯誤處理
        print(f"An error occurred during the SVM Classification training process:\n{e}")
        return None

# 函數: 使用隨機搜索的 SVM 回歸模型
def svm_regress_random_search(feature_train, target_train, param_distributions, 
                              model_random_state=42, search_random_state=42, 
                              n_iter=200, cv=5, n_jobs=-1, verbose=2):
    """
    參數: 
    - feature_train: 訓練數據集的特徵 (numpy.ndarray 或 pandas.DataFrame) 
    - target_train: 訓練數據集對應的目標值 (numpy.ndarray 或 pandas.Series)
    - param_distributions: 隨機搜索中超參數的分佈 (dict)
    - model_random_state: 初始化 SVR 的隨機種子 (int, 預設為 42)
    - search_random_state: 初始化 RandomizedSearchCV 的隨機種子 (int, 預設為 42)
    - n_iter: 抽樣的參數設置數量 (int, 預設為 200)
    - cv: 交叉驗證的折數 (int, 預設為 5)
    - n_jobs: 並行運行的作業數 (int, 預設為 -1, 使用所有處理器)
    - verbose: 日誌詳細程度 (int, 預設為 2)

    返回: 
    - 通過 RandomizedSearchCV 找到的最佳 SVM 回歸器 (SVR 物件)
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
        # 初始化 SVR
        svm = SVR(random_state=model_random_state)

        # 創建 RandomizedSearchCV 對象
        svm_random_search = RandomizedSearchCV(estimator=svm, 
                                               param_distributions=param_distributions, 
                                               random_state=search_random_state, 
                                               n_iter=n_iter, cv=cv, 
                                               n_jobs=n_jobs, verbose=verbose)

        # 對訓練數據進行擬合
        svm_random_search.fit(feature_train, target_train)

        # 獲得最佳估計器 (最佳 SVM 回歸模型)
        best_svm_regressor = svm_random_search.best_estimator_

        # 打印最佳 SVM 回歸模型的詳細資訊
        print("Best SVM Regressor Model")
        print("=" * 80)
        print("Best parameters:")
        for param, value in svm_random_search.best_params_.items():
            print(f"{param}: {value}")
        print("-" * 80)
        print(f"Validation Set Score: {svm_random_search.best_score_}")
        print("=" * 80 + "\n")

        return best_svm_regressor

    except Exception as e:
        # 錯誤處理
        print(f"An error occurred during the SVM Regression training process:\n{e}")
        return None
    