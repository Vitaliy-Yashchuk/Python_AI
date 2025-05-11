import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

X = np.linspace(-20, 20, 400).reshape(-1, 1)
y = np.sin(X).flatten() + 0.1 * (X.flatten() ** 2)

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(X, y, label='Real Function: sin(x) + 0.1x²', color='blue')
plt.plot(X, y_pred, label='Predicted by Linear Regression', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function Approximation')
plt.legend()
plt.grid(True)
plt.show()
