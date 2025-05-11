import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def time_to_minutes(hour, minute):
    return hour * 60 + minute

np.random.seed(42)
X = np.linspace(0, 1440, 1000).reshape(-1, 1) 
Y = (
    30 + 10 * np.sin((X.flatten() - 480) * (2 * np.pi / 1440)) +
    np.random.normal(0, 2, size=X.shape[0])
)

poly_model = make_pipeline(PolynomialFeatures(degree=5), LinearRegression())
poly_model.fit(X, Y)
poly_pred = poly_model.predict(X)

nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=5000, random_state=1)
nn_model.fit(X, Y)
nn_pred = nn_model.predict(X)

plt.figure(figsize=(12, 6))
plt.plot(X, Y, label='Real Data (with noise)', color='lightgray')
plt.plot(X, poly_pred, label='Polynomial Regression', color='red')
plt.plot(X, nn_pred, label='Neural Network', color='green')
plt.xlabel('Time of Day (minutes)')
plt.ylabel('Trip Duration (minutes)')
plt.title('Trip Duration Prediction by Time of Day')
plt.legend()
plt.grid(True)
plt.show()

times = [(10, 30), (0, 0), (2, 40)]
for h, m in times:
    t_min = time_to_minutes(h, m)
    poly_val = poly_model.predict(np.array([[t_min]]))[0]
    nn_val = nn_model.predict(np.array([[t_min]]))[0]
    print(f"Time {h:02d}:{m:02d} => Poly: {poly_val:.2f} min, NN: {nn_val:.2f} min")

mae_poly = mean_absolute_error(Y, poly_pred)
mse_poly = mean_squared_error(Y, poly_pred)

mae_nn = mean_absolute_error(Y, nn_pred)
mse_nn = mean_squared_error(Y, nn_pred)

print("\n--- Evaluation ---")
print(f"Polynomial Regression: MAE={mae_poly:.2f}, MSE={mse_poly:.2f}")
print(f"Neural Network:        MAE={mae_nn:.2f}, MSE={mse_nn:.2f}")
