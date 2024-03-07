from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV

# 函数: 線性回歸模型
def linear_regress(feature_train, target_train, model_random_state=42):
    """
    參數: 
    - feature_train: 訓練數據集的特徵 (numpy.ndarray 或 pandas.DataFrame) 
    - target_train: 訓練數據集對應的目標值 (numpy.ndarray 或 pandas.Series)
    - model_random_state: 初始化 LinearRegression 的隨機種子 (int, 預設為 42)

    返回: 
    - 訓練完成的線性回歸模型 (LinearRegression 物件)
      如果訓練過程中發生錯誤，則返回 None
    """

    # 驗證 feature_train 和 label_train 的長度是否匹配
    if len(feature_train) != len(target_train):
        raise ValueError("The lengths of feature_train and target_train do not match")

    try:
        # 初始化線性回歸器
        linear = LinearRegression(random_state=model_random_state)

        # 對訓練數據進行擬合
        linear.fit(feature_train, target_train)

        return linear
    
    except Exception as e:
        # 錯誤處理
        print(f"An error occurred during the Linear Regression training process:\n{e}")
        return None

# 函数: 使用隨機搜索的 Ridge 回歸模型
def ridge_regress_random_search(feature_train, target_train, param_distributions, 
                                model_random_state=42, search_random_state=42, 
                                n_iter=200, cv=5, n_jobs=-1, verbose=2):
    """
    參數: 
    - feature_train: 訓練數據集的特徵 (numpy.ndarray 或 pandas.DataFrame) 
    - target_train: 訓練數據集對應的目標值 (numpy.ndarray 或 pandas.Series)
    - param_distributions: 隨機搜索中超參數的分佈 (dict)
    - model_random_state: 初始化 Ridge 的隨機種子 (int, 預設為 42)
    - search_random_state: 初始化 RandomizedSearchCV 的隨機種子 (int, 預設為 42)
    - n_iter: 抽樣的參數設置數量 (int, 預設為 200)
    - cv: 交叉驗證的折數 (int, 預設為 5)
    - n_jobs: 並行運行的作業數 (int, 預設為 -1, 使用所有處理器)
    - verbose: 日誌詳細程度 (int, 預設為 2)

    返回: 
    - 通過 RandomizedSearchCV 找到的最佳 Ridge 回歸器 (Ridge 物件)
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
        # 初始化嶺回歸器
        ridge = Ridge(random_state=model_random_state)

        # 創建 RandomizedSearchCV 物件
        ridge_random_search = RandomizedSearchCV(estimator=ridge, 
                                                 param_distributions=param_distributions, 
                                                 random_state=search_random_state, 
                                                 n_iter=n_iter, cv=cv, 
                                                 n_jobs=n_jobs, verbose=verbose)

        # 對訓練數據進行擬合
        ridge_random_search.fit(feature_train, target_train)

        # 獲得最佳估計器 (最佳 Ridge 回歸模型)
        best_ridge_regressor = ridge_random_search.best_estimator_

        # 打印最佳 Ridge 回歸模型的詳細資訊
        print("Best Ridge Regressor Model")
        print("=" * 80)
        print("Best parameters:")
        for param, value in ridge_random_search.best_params_.items():
            print(f"{param}: {value}")
        print("-" * 80)
        print(f"Validation Set Score: {ridge_random_search.best_score_}")
        print("=" * 80 + "\n")

        return best_ridge_regressor
    
    except Exception as e:
        # 錯誤處理
        print(f"An error occurred during the Ridge Regression training process:\n{e}")
        return None

# 函数: 使用隨機搜索的 Lasso 回歸模型
def lasso_regress_random_search(feature_train, target_train, param_distributions, 
                                model_random_state=42, search_random_state=42, 
                                n_iter=200, cv=5, n_jobs=-1, verbose=2):
    """
    參數: 
    - feature_train: 訓練數據集的特徵 (numpy.ndarray 或 pandas.DataFrame) 
    - target_train: 訓練數據集對應的目標值 (numpy.ndarray 或 pandas.Series)
    - param_distributions: 隨機搜索中超參數的分佈 (dict)
    - model_random_state: 初始化 Lasso 的隨機種子 (int, 預設為 42)
    - search_random_state: 初始化 RandomizedSearchCV 的隨機種子 (int, 預設為 42)
    - n_iter: 抽樣的參數設置數量 (int, 預設為 200)
    - cv: 交叉驗證的折數 (int, 預設為 5)
    - n_jobs: 並行運行的作業數 (int, 預設為 -1, 使用所有處理器)
    - verbose: 日誌詳細程度 (int, 預設為 2)

    返回: 
    - 通過 RandomizedSearchCV 找到的最佳 Lasso 回歸器 (Lasso 物件)
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
        # 初始化 Lasso
        lasso = Lasso(random_state=model_random_state)

        # 創建 RandomizedSearchCV 對象
        lasso_random_search = RandomizedSearchCV(estimator=lasso, 
                                                 param_distributions=param_distributions, 
                                                 random_state=search_random_state, 
                                                 n_iter=n_iter, cv=cv, 
                                                 n_jobs=n_jobs, verbose=verbose)

        # 對訓練數據進行擬合
        lasso_random_search.fit(feature_train, target_train)

        # 獲得最佳估計器 (最佳 Lasso 回歸模型)
        best_lasso_regressor = lasso_random_search.best_estimator_

        # 打印最佳 Lasso 回歸模型的詳細資訊
        print("Best Lasso Regressor Model")
        print("=" * 80)
        print("Best parameters:")
        for param, value in lasso_random_search.best_params_.items():
            print(f"{param}: {value}")
        print("-" * 80)
        print(f"Validation Set Score: {lasso_random_search.best_score_}")
        print("=" * 80 + "\n")

        return best_lasso_regressor

    except Exception as e:
        # 錯誤處理
        print(f"An error occurred during the Lasso Regression training process:\n{e}")
        return None
    