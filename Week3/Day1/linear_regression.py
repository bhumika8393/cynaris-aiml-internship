import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("house_prices.csv")

print("=" * 50)
print("House Price Dataset")
print("=" * 50)
print(df)

# ==========================
# Features & Target
# ==========================

X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]

# ==========================
# Train/Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Models
# ==========================

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0)
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    mse = mean_squared_error(y_test, prediction)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    results.append([name, mse, rmse, mae, r2])

    print("\n", "=" * 40)
    print(name)

    if hasattr(model, "coef_"):
        print("Coefficients :", model.coef_)

    print("Intercept :", model.intercept_)
    print("MSE :", mse)
    print("RMSE :", rmse)
    print("MAE :", mae)
    print("R² :", r2)

# ==========================
# Save Comparison Table
# ==========================

comparison = pd.DataFrame(
    results,
    columns=["Model", "MSE", "RMSE", "MAE", "R2"]
)

comparison.to_csv("model_comparison.csv", index=False)

print("\n")
print(comparison)

# ==========================
# Plot Predicted vs Actual
# ==========================

linear = LinearRegression()
linear.fit(X_train, y_train)

pred = linear.predict(X_test)

plt.figure(figsize=(6,5))
plt.scatter(y_test, pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual")
plt.grid(True)

plt.savefig("predicted_vs_actual.png")

# ==========================
# Residual Plot
# ==========================

residuals = y_test - pred

plt.figure(figsize=(6,5))
plt.scatter(pred, residuals)

plt.axhline(y=0, color="red")

plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)

plt.savefig("residual_plot.png")

plt.show()

print("\nProject Completed Successfully!")