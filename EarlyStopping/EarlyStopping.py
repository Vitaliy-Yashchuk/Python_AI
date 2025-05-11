import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(42)
days = np.arange(1, 366).reshape(-1, 1)
energy = (
    200 + 50 * np.sin((2 * np.pi * (days.flatten() - 80)) / 365) +  # річна сезонність
    np.random.normal(0, 10, size=days.shape[0])  # шум
)

df = pd.DataFrame({'DayOfYear': days.flatten(), 'EnergyConsumption': energy})
X = df[['DayOfYear']]
y = df['EnergyConsumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label='Actual', color='blue')
plt.scatter(X_test, y_pred, label='Predicted', color='red', alpha=0.7)
plt.xlabel('Day of Year')
plt.ylabel('Energy Consumption')
plt.title('Electricity Consumption Prediction')
plt.legend()
plt.grid(True)
plt.show()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

days_to_predict = pd.DataFrame({'DayOfYear': [1, 166, 359]})
predictions = model.predict(days_to_predict)
for d, p in zip(days_to_predict['DayOfYear'], predictions):
    print(f"Day {d}: Predicted energy consumption = {p:.2f}")
