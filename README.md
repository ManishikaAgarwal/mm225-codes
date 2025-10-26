#lab5 
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




# Exercise 2: Linear regression with scikit-learn for Arrhenius fit
# Model: ln(rate) = ln(A) - (Q/R)*(1/T)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

R = 8.314  # J/mol·K

# 1) Load data (columns must be: Temperature, Rate)
df = pd.read_csv("rate.csv")
T = df["Temperature"].to_numpy()
rate = df["Rate"].to_numpy()

# 2) Transform to linear form: Y = ln(rate), X = 1/T
X = (1.0 / T).reshape(-1, 1)   # predictor needs shape (n,1)
Y = np.log(rate)               # target

# 3) Fit linear model: Y = beta0 + beta1 * X
model = LinearRegression()
model.fit(X, Y)

beta0 = model.intercept_
beta1 = model.coef_[0]

# 4) Convert back to Arrhenius parameters
A = np.exp(beta0)
Q = -beta1 * R

print(f"Intercept (beta0) = {beta0:.6f}")
print(f"Slope (beta1)     = {beta1:.6f}")
print(f"Estimated A        = {A:.6e}")
print(f"Estimated Q (J/mol)= {Q:.3f}")

# 5) Predictions in original (rate) scale
Y_pred = model.predict(X)
rate_pred = np.exp(Y_pred)

# 6) Simple error metrics
MSE  = np.mean((rate - rate_pred)**2)
RMSE = np.sqrt(MSE)
MAE  = np.mean(np.abs(rate - rate_pred))
MAPE = np.mean(np.abs((rate - rate_pred) / rate)) * 100

print(f"MSE  = {MSE:.6e}")
print(f"RMSE = {RMSE:.6e}")
print(f"MAE  = {MAE:.6e}")
print(f"MAPE = {MAPE:.3f}%")

# 7) Plot: experimental points vs fitted curve
order = np.argsort(T)
T_s, rate_pred_s = T[order], rate_pred[order]

plt.scatter(T, rate, label="Experimental data")
plt.plot(T_s, rate_pred_s, label="Fitted (sklearn)", linewidth=2)
plt.xlabel("Temperature (K)")
plt.ylabel("Rate")
plt.legend()
plt.tight_layout()
plt.show()




#exercise2
# Exercise 2: Linear regression with scikit-learn for Arrhenius fit
# Model: ln(rate) = ln(A) - (Q/R)*(1/T)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

R = 8.314  # J/mol·K

# 1) Load data (columns must be: Temperature, Rate)
df = pd.read_csv("rate.csv")
T = df["Temperature"].to_numpy()
rate = df["Rate"].to_numpy()

# 2) Transform to linear form: Y = ln(rate), X = 1/T
X = (1.0 / T).reshape(-1, 1)   # predictor needs shape (n,1)
Y = np.log(rate)               # target

# 3) Fit linear model: Y = beta0 + beta1 * X
model = LinearRegression()
model.fit(X, Y)

beta0 = model.intercept_
beta1 = model.coef_[0]

# 4) Convert back to Arrhenius parameters
A = np.exp(beta0)
Q = -beta1 * R

print(f"Intercept (beta0) = {beta0:.6f}")
print(f"Slope (beta1)     = {beta1:.6f}")
print(f"Estimated A        = {A:.6e}")
print(f"Estimated Q (J/mol)= {Q:.3f}")

# 5) Predictions in original (rate) scale
Y_pred = model.predict(X)
rate_pred = np.exp(Y_pred)

# 6) Simple error metrics
MSE  = np.mean((rate - rate_pred)**2)
RMSE = np.sqrt(MSE)
MAE  = np.mean(np.abs(rate - rate_pred))
MAPE = np.mean(np.abs((rate - rate_pred) / rate)) * 100

print(f"MSE  = {MSE:.6e}")
print(f"RMSE = {RMSE:.6e}")
print(f"MAE  = {MAE:.6e}")
print(f"MAPE = {MAPE:.3f}%")

# 7) Plot: experimental points vs fitted curve
order = np.argsort(T)
T_s, rate_pred_s = T[order], rate_pred[order]

plt.scatter(T, rate, label="Experimental data")
plt.plot(T_s, rate_pred_s, label="Fitted (sklearn)", linewidth=2)
plt.xlabel("Temperature (K)")
plt.ylabel("Rate")
plt.legend()
plt.tight_layout()
plt.show()

#exercise3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Load data
df = pd.read_csv("cp.csv")   # file should have: Temperature, Cp
T = df["Temperature"].to_numpy(dtype=float)
Cp = df["Cp"].to_numpy(dtype=float)

# 2) Quadratic fit: Cp = a + b T + c T^2
deg = 2
coeffs = np.polyfit(T, Cp, deg=deg)   # returns [c, b, a]
c, b, a = coeffs
print(f"a = {a:.6f}, b = {b:.6e}, c = {c:.6e}")

# 3) Predictions and metrics
Cp_pred = np.polyval(coeffs, T)

MSE  = np.mean((Cp - Cp_pred)**2)
RMSE = np.sqrt(MSE)
MAE  = np.mean(np.abs(Cp - Cp_pred))
MAPE = np.mean(np.abs((Cp - Cp_pred)/Cp)) * 100

print(f"MSE  = {MSE:.6e}")
print(f"RMSE = {RMSE:.6e}")
print(f"MAE  = {MAE:.6e}")
print(f"MAPE = {MAPE:.3f}%")

# Optional: R^2
ss_res = np.sum((Cp - Cp_pred)**2)
ss_tot = np.sum((Cp - np.mean(Cp))**2)
R2 = 1 - ss_res/ss_tot
print(f"R^2  = {R2:.5f}")

# 4) Pretty plot (sorted for smooth line)
order = np.argsort(T)
T_s, Cp_pred_s = T[order], Cp_pred[order]

plt.scatter(T, Cp, label="Experimental Cp")
plt.plot(T_s, Cp_pred_s, linewidth=2, label="Quadratic fit")
plt.xlabel("Temperature (K)")
plt.ylabel("Cp")
plt.legend()
plt.tight_layout()
plt.show()

# 5) Evaluate Cp and dCp/dT at any T0
T0 = float(input("Enter a temperature (K) to evaluate Cp: ") or 300)
Cp_T0 = a + b*T0 + c*T0*T0
dCp_dT_T0 = b + 2*c*T0
print(f"Cp({T0} K)      = {Cp_T0:.6f}")
print(f"dCp/dT({T0} K)  = {dCp_dT_T0:.6f}")


