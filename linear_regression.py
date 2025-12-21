# Prediecting Fibonacci numbers

import numpy as np
from sklearn.linear_model import LinearRegression

# Fibonacci numbers for n = 1 to 10
fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

# Prepare data
n = np.arange(1, len(fib) + 1).reshape(-1, 1)  # feature: n
print("n",n)
y = np.array(fib)  # target: Fibonacci value
print("y",y)

# Train linear regression
model = LinearRegression()
model.fit(n, y)

#print("Slope:", model.coef_[0])
#print("Intercept:", model.intercept_)

# Predict the next Fibonacci number (n = 11)
n_new = np.array([[11]])
pred = model.predict(n_new)
print("pred",pred)
print("Predicted value at n=11:", pred[0])
#linear regression is not suitable for predicting Fibonacci numbers, 
#As it assumes a linear relationship between the input and output variables, which is not the case for Fibonacci numbers. 
#The Fibonacci sequence is defined by a recursive formula, and the relationship between consecutive Fibonacci numbers is not linear.