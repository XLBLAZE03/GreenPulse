# excel_ml_pipeline.py

# --- Step 1: Import libraries ---
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --- Step 2: Load csv data ---
# Replace 'data .csv' with the path to your csv file
df = pd.read_csv(("C:\lifepulse\GreenPulse\IṁṇṇNEOS_Daily_CO2_Dataset_2010_2024.csv"))

print("Preview of data:")
print(df.head())

# --- Step 3: Preprocess data --
# Drop rows with missing values
df = df.dropna()

# Convert categorical columns to numeric (dummy encoding)
df = pd.get_dummies(df, drop_first=True)

print("\nData after preprocessing:")
print(df.head())

# --- Step 4: Visualization ---
# Plot the first two numeric columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

if len(numeric_cols) >= 2:
    x_col = numeric_cols[0]
    y_col = numeric_cols[1]

    plt.figure(figsize=(8,6))
    plt.scatter(df[x_col], df[y_col], color="blue", alpha=0.7)
    plt.title(f"{y_col} vs {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.show()
else:
    print("Not enough numeric columns to visualize.")

# --- Step 5: Train a simple ML model ---
if len(numeric_cols) > 1:
    target = numeric_cols[-1]
    features = [col for col in numeric_cols[:-1]]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"\nModel trained to predict '{target}'")
    print("Accuracy (R² score):", score)

    # Plot predictions vs actual
    y_pred = model.predict(X_test)
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, color="green", alpha=0.7)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Prediction Performance for {target}")
    plt.grid(True)
    plt.show()
else:
    print("Not enough numeric columns to train a model.")
