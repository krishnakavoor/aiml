# Prediecting Fibonacci numbers

import numpy as np
from sklearn.linear_model import LinearRegression

# numbers for n = 1 to 10
num = [0,1, 2, 3, 4, 5, 6, 7, 8, 9,10]

# Prepare data
n = np.arange(1, len(num) + 1).reshape(-1, 1)  # feature: n
print("n",n)
y = np.array(num)  # target: Fibonacci value
print("y",y)

# Train linear regression
model = LinearRegression()
model.fit(n, y)

#print("Slope:", model.coef_[0])
#print("Intercept:", model.intercept_)

# Predict the next number (n = with couting it)
n_new = np.array([[len(num)+1]])
pred = model.predict(n_new)
print("pred",pred)
print("Predicted value at n=11:", pred[0])
#linear regression is not suitable for predicting Fibonacci numbers, 
#As it assumes a linear relationship between the input and output variables, which is not the case for Fibonacci numbers. 
#The Fibonacci sequence is defined by a recursive formula, and the relationship between consecutive Fibonacci numbers is not linear.