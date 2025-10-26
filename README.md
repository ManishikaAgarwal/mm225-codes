#lab6
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

#LAB7 

import pandas as pd
import statsmodels.api as sm

# Load data
data = pd.read_csv("data.csv")

# Dependent variable
y = data['y']

# Independent variables (all except 'y')
X = data.drop(columns=['y'])

selected_features = []                 # features chosen so far
remaining_features = list(X.columns)   # features we can still try

alpha = 0.05   # significance threshold for adding a feature

while len(remaining_features) > 0:
    p_values = {}  # store p-value of the "new" feature for each candidate
    
    for feature in remaining_features:
        # Try a model that adds just this one new feature to those already selected
        trial_features = selected_features + [feature]
        X_trial = sm.add_constant(X[trial_features])  # add intercept
        model = sm.OLS(y, X_trial).fit()             # fit OLS
        
        # p-value of the candidate feature *in this trial model*
        p_values[feature] = model.pvalues[feature]
        
    # Pick the candidate with the lowest p-value this round
    best_feature = min(p_values, key=p_values.get)
    
    # If it is statistically significant, accept it and continue; else stop
    if p_values[best_feature] < alpha:
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        print(f"Added: {best_feature} (p-value = {p_values[best_feature]:.4f})")
    else:
        print("No more statistically significant features to add.")
        break

print("\nFinal selected features:", selected_features)

# Fit and show the final model with only the selected features
X_final = sm.add_constant(X[selected_features])
final_model = sm.OLS(y, X_final).fit()
print("\nFinal Model Summary:")
print(final_model.summary())


#LAB8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Load dataset
data = pd.read_csv("data2.csv")

# y = first column, X = remaining columns
y = data.iloc[:, 0]        # dependent variable
X = data.iloc[:, 1:]       # independent variables x1, x2, x3

# Perform one random train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Print the learned coefficients
print("Intercept (b0) =", model.intercept_)
print("Coefficients (b1, b2, b3) =", model.coef_)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate errors
MSE = np.mean((y_test - y_pred)**2)
print("Test MSE =", MSE)


#LOGISTIC REGRESSION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("oring.csv")

# Temperature is X, Failure (0/1) is y
X = data[['Temperature']]
y = data['Failure']

# Fit logistic regression model
model = LogisticRegression(solver='lbfgs')
model.fit(X, y)

# Predicted probability of failure for each temperature
prob = model.predict_proba(X)[:, 1]

print("Intercept =", model.intercept_)
print("Coefficient =", model.coef_)

# Plot data + logistic curve
plt.scatter(X, y, label="Actual Failure Data")
plt.plot(X, prob, label="Predicted Failure Probability", color='red')
plt.xlabel("Temperature (°F)")
plt.ylabel("Probability of Failure")
plt.legend()
plt.show()

#LAB9

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv("data3.csv")

# y is first column, X is next 5 columns
y = data.iloc[:, 0]
X = data.iloc[:, 1:]

# Split data into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### 1) Least Squares (OLS)
ols = LinearRegression().fit(X_train, y_train)
y_pred_ols = ols.predict(X_test)
print("\nOLS Coefficients:", ols.coef_)
print("OLS Test MSE:", mean_squared_error(y_test, y_pred_ols))

### 2) Lasso (manual alpha)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
print("\nLasso Coefficients:", lasso.coef_)
print("Lasso Test MSE:", mean_squared_error(y_test, y_pred_lasso))

### 3) LassoCV (auto alpha search)
lasso_cv = LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100]).fit(X_train, y_train)
y_pred_lasso_cv = lasso_cv.predict(X_test)
print("\nLassoCV Best Alpha:", lasso_cv.alpha_)
print("LassoCV Coefficients:", lasso_cv.coef_)
print("LassoCV Test MSE:", mean_squared_error(y_test, y_pred_lasso_cv))

### 4) Ridge (manual alpha)
ridge = Ridge(alpha=1.0).fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print("\nRidge Coefficients:", ridge.coef_)
print("Ridge Test MSE:", mean_squared_error(y_test, y_pred_ridge))

### 5) RidgeCV (auto alpha search)
ridge_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100]).fit(X_train, y_train)
y_pred_ridge_cv = ridge_cv.predict(X_test)
print("\nRidgeCV Best Alpha:", ridge_cv.alpha_)
print("RidgeCV Coefficients:", ridge_cv.coef_)
print("RidgeCV Test MSE:", mean_squared_error(y_test, y_pred_ridge_cv))




