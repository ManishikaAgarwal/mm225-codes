import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Universal gas constant
R = 8.314  

# Read CSV
data = pd.read_csv("rate.csv")   # Ensure the file is in the same folder

T = data['Temperature'].values
rate = data['Rate'].values

# Transform variables
X = 1 / T
Y = np.log(rate)

# Construct matrix X (with intercept column)
X_mat = np.column_stack((np.ones(len(X)), X))

# Apply matrix formula: beta = (X^T X)^(-1) X^T Y
beta = np.linalg.inv(X_mat.T @ X_mat) @ (X_mat.T @ Y)

beta0, beta1 = beta[0], beta[1]

# Extract A and Q
A = np.exp(beta0)
Q = -beta1 * R

print("Estimated A =", A)
print("Estimated Q =", Q)

# Predicted Y
Y_pred = X_mat @ beta
rate_pred = np.exp(Y_pred)

# Error Metrics
MSE = np.mean((rate - rate_pred)**2)
RMSE = np.sqrt(MSE)
MAE = np.mean(np.abs(rate - rate_pred))
MAPE = np.mean(np.abs((rate - rate_pred) / rate)) * 100

print("MSE =", MSE)
print("RMSE =", RMSE)
print("MAE =", MAE)
print("MAPE (%) =", MAPE)

# Plot
plt.scatter(T, rate, label='Experimental Data')
plt.plot(T, rate_pred, label='Fitted Line', linewidth=2)
plt.xlabel("Temperature (K)")
plt.ylabel("Rate")
plt.legend()
plt.show()

