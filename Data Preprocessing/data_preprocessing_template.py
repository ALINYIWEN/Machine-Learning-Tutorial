# 數據預處理

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer # 特徵縮放有兩種

# 載入 dataset  
dir_data = './data/'
data = os.path.join(dir_data, 'customer_data.csv')
dataset = pd.read_csv(data)
X = dataset.iloc[:, :-1].values #[row, column] 去除target那column
y = dataset.iloc[:, 3].values   # 取Target column

# 處理 missing data
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
X[:,1:3] = imputer.fit_transform(X[:,1:3])

# Encoding categorical data 轉換無意義的資料為有意義的數值
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:, 0]) # 同時做擬和與轉換
one_hot_encoder = ColumnTransformer([("Country",OneHotEncoder(),[0])], remainder= 'passthrough') # 選擇要處理的是哪個column
X = one_hot_encoder.fit_transform(X)

labelencoder_y = LabelEncoder()      # 將 Target 轉換為數值
y = labelencoder_y.fit_transform(y)  # 同時做擬和與轉換

# 將數據集分為訓練集和測試集: test_size: 測試集的占比, random_state: 隨機選取數據的機率 0=一樣的結果
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

# 特徵縮放 Feature Scaling : 使數據的量級靠近, 避免部分數據影響過大 
# 分類問題不需對 y 因變量做特徵縮放 , 回歸問題看情況做 
Scaling_X = StandardScaler()
X_train = Scaling_X.fit_transform(X_train)
X_test = Scaling_X.transform(X_test)  