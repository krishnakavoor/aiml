import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Fibonacci sequence values for n = 1 to 10
fib = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34])

# Feature: n values
n = np.arange(1, len(fib) + 1).reshape(-1, 1)

# Polynomial degree (try 2, 3, or 4)
degree = 6
poly = PolynomialFeatures(degree=degree)
n_poly = poly.fit_transform(n)

# Fit model
model = LinearRegression()
model.fit(n_poly, fib)

print(f"Coefficients (degree={degree}):", model.coef_)
print("Intercept:", model.intercept_)

# Predict next Fibonacci value n = 11
n_new = np.array([[len(fib) + 1]])
n_new_poly = poly.transform(n_new)
pred = model.predict(n_new_poly)

print("Predicted Fibonacci value at n=11:", pred[0])
