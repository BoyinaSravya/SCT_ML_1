import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==================================================
# HOUSE PRICE PREDICTION USING LINEAR REGRESSION
# ==================================================

print("=" * 60)
print("HOUSE PRICE PREDICTION USING LINEAR REGRESSION")
print("=" * 60)

# ==================================================
# LOAD DATASET
# ==================================================

dataset_path = r"C:\Users\DELL\Downloads\house-prices-advanced-regression-techniques\train.csv"

df = pd.read_csv(dataset_path)

print("\nDataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# ==================================================
# SELECT FEATURES
# ==================================================

features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
target = "SalePrice"

data = df[features + [target]].dropna()

X = data[features]
y = data[target]

# ==================================================
# CREATE GRAPH FOLDER
# ==================================================

os.makedirs("graphs", exist_ok=True)

# ==================================================
# GRAPH 1 - CORRELATION HEATMAP
# ==================================================

plt.figure(figsize=(6,4))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("graphs/correlation_heatmap.png")
plt.show()

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==================================================
# TRAIN MODEL
# ==================================================

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ==================================================
# PREDICTIONS
# ==================================================

y_pred = model.predict(X_test)

# ==================================================
# MODEL EVALUATION
# ==================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-" * 30)
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 4))

# ==================================================
# GRAPH 2 - PRICE DISTRIBUTION
# ==================================================

plt.figure(figsize=(6,4))
sns.histplot(data["SalePrice"], kde=True)
plt.title("House Price Distribution")
plt.tight_layout()
plt.savefig("graphs/price_distribution.png")
plt.show()

# ==================================================
# GRAPH 3 - ACTUAL VS PREDICTED
# ==================================================

plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")

plt.tight_layout()
plt.savefig("graphs/actual_vs_predicted.png")
plt.show()

# ==================================================
# SAVE MODEL
# ==================================================

joblib.dump(model, "house_price_model.pkl")

print("\nModel Saved Successfully!")

# ==================================================
# USER PREDICTION
# ==================================================

print("\n" + "=" * 60)
print("HOUSE PRICE PREDICTION SYSTEM")
print("=" * 60)

sqft = float(input("Enter Square Footage: "))
bedrooms = int(input("Enter Number of Bedrooms: "))
bathrooms = int(input("Enter Number of Bathrooms: "))

user_data = pd.DataFrame({
    "GrLivArea": [sqft],
    "BedroomAbvGr": [bedrooms],
    "FullBath": [bathrooms]
})

prediction = model.predict(user_data)

print("\nPredicted House Price: ₹{:,.2f}".format(prediction[0]))

print("\nProject Executed Successfully!")