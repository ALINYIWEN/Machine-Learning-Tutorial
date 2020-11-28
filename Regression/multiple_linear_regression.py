# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.regression.linear_model as sm
from sklearn.compose import ColumnTransformer

# Importing the dataset
dataset = pd.read_csv('./data/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) # X[: "要處理的那 Column"]
one_hot_encoder = ColumnTransformer([("State", OneHotEncoder(),[3])], remainder= 'passthrough') # 選擇要處理的是哪個column
X = one_hot_encoder.fit_transform(X)

# Avoiding the Dummy Variable Trap  避免有三個column的值和為1
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_predict = regressor.predict(X_test)

# 使用反向淘汰 Backward Elimination 刪除不需要的自變量來最佳化模型
X_train = np.append(arr = np.ones((40, 1)).astype(int), values = X_train, axis = 1) # 新增一個COLUMN在 X_train裡
# 第一步 設定 significance level 來篩選
# 第二步 輸入所有的自變量來擬和模型
X_optimal = X_train [:, [0, 1, 2, 3, 4, 5]].astype(int)     # 沒加.astype(int)會出錯: X_opt array has a dtype of object and this may be causing an error.                     
regressor_OLS = sm.OLS(endog = y_train, exog = X_optimal).fit() # OLS: 簡單的普通最少平方模型。
regressor_OLS.summary()  # Summary 能讓我們查看回歸訊息 用來挑過大的 P Value
# 發現第　index 2 column不需要，因此拿掉
X_optimal = X_train [:, [0, 1, 3, 4, 5]].astype(int) 
regressor_OLS = sm.OLS(endog = y_train, exog = X_optimal).fit()
regressor_OLS.summary()
X_optimal = X_train [:, [0, 3, 4, 5]].astype(int) 
regressor_OLS = sm.OLS(endog = y_train, exog = X_optimal).fit()
regressor_OLS.summary()
X_optimal = X_train [:, [0, 3, 5]].astype(int) 
regressor_OLS = sm.OLS(endog = y_train, exog = X_optimal).fit()
regressor_OLS.summary()
X_optimal = X_train [:, [0, 3]].astype(int) 
regressor_OLS = sm.OLS(endog = y_train, exog = X_optimal).fit()
regressor_OLS.summary()
print(regressor_OLS.summary())

