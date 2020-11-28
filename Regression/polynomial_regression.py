# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv('./data/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
linear_regression = LinearRegression()
linear_regression.fit(X, y)              # 直接針對原始數據擬和，因為數據太少，不須分訓練測試

# Fitting Polynomial Regression to the dataset
polynomial_regression = PolynomialFeatures(degree = 4)
X_polynomial = polynomial_regression.fit_transform(X)
linear_regression_2 = LinearRegression()
linear_regression_2.fit(X_polynomial, y) # 針對回歸後的X來做線性回歸

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regression.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, linear_regression_2.predict(polynomial_regression.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# 預測一個新的結果 (線性回歸)
predict_data = 6.5
predict_data = np.array(predict_data).reshape(1, -1)
linear_result = linear_regression.predict(predict_data) #直接輸入想預測的結果 

# 預測一個新的結果 (多項式回歸)
polynomial_result = linear_regression_2.predict(polynomial_regression.fit_transform(predict_data))