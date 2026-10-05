# ============================================================
# INDIA GDP ANALYSIS AND PREDICTION
# ============================================================

# 1. DATASET USED
# ------------------------------------------------------------
# Dataset: India GDP Dataset
# Source: World Bank
# Country: India
# Indicator: GDP (Current US$)
# The dataset contains yearly GDP values of India.


# 2. DATA PREPROCESSING
# ------------------------------------------------------------

# 2.1 DATA CLEANING
# - Select only the data for India.
# - Remove unnecessary columns.
# - Convert Year and GDP values into numeric format.
# - Remove missing or invalid values.
# - Remove duplicate years.
# - Sort the data in chronological order.


# 2.2 DATA INTEGRATION
# - The required India GDP information is selected and organized
#   from the World Bank dataset.
# - No separate external dataset is merged with this dataset.


# 2.3 DATA TRANSFORMATION
# - Convert the original wide-format data into a time-series format.
# - Convert GDP into Billion USD and Trillion USD.
# - Calculate Year-on-Year GDP Growth Percentage.
# - Apply logarithmic transformation to GDP.
# - Convert the Year column into the required numeric format.


# 2.4 DATA REDUCTION
# - Remove unnecessary columns such as:
#   Country Name
#   Country Code
#   Indicator Name
#   Indicator Code
# - Only the required information is retained for analysis.


# 3. METHODOLOGY / ALGORITHM / MODEL IMPLEMENTED
# ------------------------------------------------------------
# The project follows these steps:
#
# World Bank GDP Dataset
#          ↓
# Data Preprocessing
#          ↓
# Training and Testing Data Split
#          ↓
# Machine Learning Models
#          ↓
# GDP Prediction
#          ↓
# Model Evaluation
#          ↓
# Model Comparison
#
# Three regression algorithms are implemented:
#
# 1. Linear Regression
# 2. Decision Tree Regression
# 3. Random Forest Regression
#
# Input Feature:
# - Year
#
# Target Variable:
# - GDP
#
# Problem Type:
# - Regression


# 4. TRAINING / TESTING PROCESS
# ------------------------------------------------------------
# - The dataset is divided into training and testing data.
# - The last 10 years of data are used for testing.
# - The remaining historical data are used for training.
#
# Training:
# - The models learn the relationship between Year and GDP
#   using the training data.
#
# Testing:
# - The trained models predict GDP values for the unseen
#   testing years.
# - The predicted values are compared with the actual GDP values.


# 5. RESULTS OBTAINED
# ------------------------------------------------------------
# - GDP values are predicted using all three regression models.
# - The actual GDP values and predicted GDP values are compared.
# - The performance of Linear Regression, Decision Tree Regression,
#   and Random Forest Regression is compared.
# - The model with better evaluation scores can be considered
#   the better-performing model for this dataset.


# 6. EVALUATION METRICS
# ------------------------------------------------------------
# The following evaluation metrics are used:
#
# 1. MAE (Mean Absolute Error)
#    - Measures the average absolute difference between
#      actual and predicted GDP values.
#    - Lower MAE indicates better performance.
#
# 2. RMSE (Root Mean Squared Error)
#    - Measures prediction error and gives more importance
#      to larger errors.
#    - Lower RMSE indicates better performance.
#
# 3. MAPE (Mean Absolute Percentage Error)
#    - Measures the average percentage error between
#      actual and predicted values.
#    - Lower MAPE indicates better performance.
#
# 4. R2 Score (R-Squared)
#    - Measures how well the model explains the variation
#      in GDP values.
#    - Higher R2 Score indicates better performance.


# 7. GRAPHS / VISUALIZATIONS
# ------------------------------------------------------------
# The following graphs are generated:
#
# 1. India GDP Trend
# 2. Training and Testing of India GDP
# 3. Actual vs Predicted India GDP
# 4. Year-on-Year GDP Growth
# 5. MAE Comparison
# 6. RMSE Comparison
# 7. MAPE Comparison
# 8. R2 Score Comparison
#
# These visualizations help to understand the GDP trend,
# model predictions, and performance of different algorithms.


# ============================================================
# END OF PROJECT DESCRIPTION
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES AND LOAD DATASET
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


print("\n============================================================")
print("              INDIA GDP MINI PROJECT")
print("============================================================")


# ============================================================
# CSV FILE PATH
# ============================================================

CSV_FILE = r"C:\Users\rahul kumar\OneDrive\PROJECT\5thsem\DWDM\indiaGDP.csv"




# ============================================================
# DATASET LOADING
# ============================================================

print("\n========== DATASET LOADING ==========")
print(CSV_FILE)


# World Bank CSV contains metadata before actual header
with open(CSV_FILE, encoding="utf-8-sig") as f:
    lines = f.readlines()


# Find actual header row
header_line = next(
    i for i, line in enumerate(lines)
    if "Country Name" in line
)


# Read CSV
raw = pd.read_csv(
    CSV_FILE,
    skiprows=header_line
)


# Remove unnecessary unnamed columns
raw = raw.loc[
    :,
    ~raw.columns.astype(str).str.startswith("Unnamed")
]


print("\nDataset loaded successfully.")
print("Dataset shape:", raw.shape)

print("\nFirst 5 rows:")
print(raw.head())


# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

print("\n============================================================")
print("                 2. DATA PREPROCESSING")
print("============================================================")


# ------------------------------------------------------------
# Step 1: Select India
# ------------------------------------------------------------

india = raw[
    raw["Country Code"].astype(str).str.upper() == "IND"
].copy()


if india.empty:
    raise ValueError(
        "India data with Country Code 'IND' was not found."
    )


print("\n1. India data selected.")
print("Shape before preprocessing:", india.shape)


# ------------------------------------------------------------
# Step 2: Remove unnecessary columns
# ------------------------------------------------------------

columns_to_remove = [
    "Country Name",
    "Country Code",
    "Indicator Name",
    "Indicator Code"
]

columns_to_remove = [
    col for col in columns_to_remove
    if col in india.columns
]

india = india.drop(
    columns=columns_to_remove
)


print("\n2. Unnecessary columns removed.")


# ------------------------------------------------------------
# Step 3: Convert wide data into time-series format
# ------------------------------------------------------------

df = india.T

df.columns = ["GDP"]

print("\n3. Dataset converted into time-series format.")


# ------------------------------------------------------------
# Step 4: Convert year to numeric
# ------------------------------------------------------------

df.index = pd.to_numeric(
    df.index,
    errors="coerce"
)

print("Year converted to numeric.")


# ------------------------------------------------------------
# Step 5: Convert GDP to numeric
# ------------------------------------------------------------

df["GDP"] = pd.to_numeric(
    df["GDP"],
    errors="coerce"
)

print("GDP converted to numeric.")


# ------------------------------------------------------------
# Step 6: Remove missing / invalid values
# ------------------------------------------------------------

before_rows = len(df)

df = df.dropna(
    subset=["GDP"]
)

df = df[
    df.index.notna()
]

df = df[
    df["GDP"] > 0
]

after_rows = len(df)

print("\n4. Missing/invalid values removed.")
print("Rows removed:", before_rows - after_rows)


# ------------------------------------------------------------
# Step 7: Convert year index to integer
# ------------------------------------------------------------

df.index = df.index.astype(int)


# ------------------------------------------------------------
# Step 8: Remove duplicate years
# ------------------------------------------------------------

duplicates = df.index.duplicated().sum()

df = df[
    ~df.index.duplicated(
        keep="first"
    )
]

print("\n5. Duplicate years removed:", duplicates)


# ------------------------------------------------------------
# Step 9: Sort by year
# ------------------------------------------------------------

df = df.sort_index()

print("Data sorted chronologically.")


# ------------------------------------------------------------
# Step 10: Check missing values
# ------------------------------------------------------------

print(
    "\n6. Missing GDP values:",
    df["GDP"].isna().sum()
)


# ------------------------------------------------------------
# Step 11: Create useful features
# ------------------------------------------------------------

# GDP in Billion USD
df["GDP_Billion_USD"] = (
    df["GDP"] / 1e9
)

# GDP in Trillion USD
df["GDP_Trillion_USD"] = (
    df["GDP"] / 1e12
)

# Year-on-Year GDP Growth
df["Growth_Percentage"] = (
    df["GDP"].pct_change() * 100
)

# Log GDP
df["Log_GDP"] = np.log(
    df["GDP"]
)


# ------------------------------------------------------------
# Display preprocessed data
# ------------------------------------------------------------

print("\n========== AFTER PREPROCESSING ==========")

print(
    df.head(10).round(2)
)

print(
    "\nFinal dataset shape:",
    df.shape
)

print("\nData types:")
print(df.dtypes)

print("\nSummary statistics:")
print(
    df.describe().round(2)
)


# ============================================================
# 3. TRAINING / TESTING PROCESS
# ============================================================

print("\n============================================================")
print("              3. TRAINING / TESTING PROCESS")
print("============================================================")


# Last 10 years = Testing
# Remaining years = Training

TEST_SIZE = 10


if len(df) <= TEST_SIZE:
    raise ValueError(
        "Dataset does not contain enough years "
        "for 10-year testing."
    )


train = df.iloc[:-TEST_SIZE].copy()

test = df.iloc[-TEST_SIZE:].copy()


print("\nTraining samples:", len(train))
print("Testing samples :", len(test))


print(
    "\nTraining period:",
    int(train.index.min()),
    "-",
    int(train.index.max())
)

print(
    "Testing period :",
    int(test.index.min()),
    "-",
    int(test.index.max())
)


# ------------------------------------------------------------
# Training data
# ------------------------------------------------------------

X_train = train.index.values.reshape(
    -1, 1
)

y_train = train["GDP"].values


# ------------------------------------------------------------
# Testing data
# ------------------------------------------------------------

X_test = test.index.values.reshape(
    -1, 1
)

y_test = test["GDP"].values


# ============================================================
# ALGORITHM 1: LINEAR REGRESSION
# ============================================================

print("\n============================================================")
print("             ALGORITHM 1: LINEAR REGRESSION")
print("============================================================")


linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_pred = linear_model.predict(
    X_test
)

print("Linear Regression training completed.")
print("Linear Regression prediction completed.")


# ============================================================
# ALGORITHM 2: DECISION TREE REGRESSION
# ============================================================

print("\n============================================================")
print("          ALGORITHM 2: DECISION TREE REGRESSION")
print("============================================================")


decision_tree_model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

decision_tree_model.fit(
    X_train,
    y_train
)

decision_tree_pred = decision_tree_model.predict(
    X_test
)

print("Decision Tree training completed.")
print("Decision Tree prediction completed.")


# ============================================================
# ALGORITHM 3: RANDOM FOREST REGRESSION
# ============================================================

print("\n============================================================")
print("          ALGORITHM 3: RANDOM FOREST REGRESSION")
print("============================================================")


random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_pred = random_forest_model.predict(
    X_test
)

print("Random Forest training completed.")
print("Random Forest prediction completed.")


# ============================================================
# 4. RESULTS OBTAINED
# ============================================================

print("\n============================================================")
print("                  4. RESULTS OBTAINED")
print("============================================================")


# ------------------------------------------------------------
# Create result table
# ------------------------------------------------------------

result = pd.DataFrame({

    "Year":
        test.index.astype(int),

    "Actual GDP (Billion USD)":
        y_test / 1e9,

    "Linear Regression":
        linear_pred / 1e9,

    "Decision Tree":
        decision_tree_pred / 1e9,

    "Random Forest":
        random_forest_pred / 1e9
})


print("\nActual vs Predicted GDP:")
print(
    result.round(2).to_string(index=False)
)


print("\nResults Obtained:")

print(
    "Three machine learning algorithms were applied "
    "to India GDP data."
)

print(
    "The last 10 years were used as testing data."
)

print(
    "GDP predictions were generated using "
    "Linear Regression, Decision Tree Regression "
    "and Random Forest Regression."
)


# ============================================================
# 5. EVALUATION METRICS
# ============================================================

print("\n============================================================")
print("                 5. EVALUATION METRICS")
print("============================================================")


# ------------------------------------------------------------
# Function to calculate metrics
# ------------------------------------------------------------

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mape = np.mean(
        np.abs(
            (actual - predicted) / actual
        )
    ) * 100

    r2 = r2_score(
        actual,
        predicted
    )

    return (
        mae / 1e9,
        rmse / 1e9,
        mape,
        r2
    )


# ------------------------------------------------------------
# Calculate metrics
# ------------------------------------------------------------

linear_mae, linear_rmse, linear_mape, linear_r2 = (
    calculate_metrics(
        y_test,
        linear_pred
    )
)

tree_mae, tree_rmse, tree_mape, tree_r2 = (
    calculate_metrics(
        y_test,
        decision_tree_pred
    )
)

forest_mae, forest_rmse, forest_mape, forest_r2 = (
    calculate_metrics(
        y_test,
        random_forest_pred
    )
)


# ------------------------------------------------------------
# Model comparison table
# ------------------------------------------------------------

metrics = pd.DataFrame({

    "Algorithm": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "MAE (Billion USD)": [
        linear_mae,
        tree_mae,
        forest_mae
    ],

    "RMSE (Billion USD)": [
        linear_rmse,
        tree_rmse,
        forest_rmse
    ],

    "MAPE (%)": [
        linear_mape,
        tree_mape,
        forest_mape
    ],

    "R2 Score": [
        linear_r2,
        tree_r2,
        forest_r2
    ]
})


print("\n========== MODEL COMPARISON ==========")

print(
    metrics.round(2).to_string(index=False)
)


# ------------------------------------------------------------
# Individual metric results
# ------------------------------------------------------------

print("\n========== LINEAR REGRESSION ==========")

print(
    "MAE  :",
    round(linear_mae, 2),
    "Billion USD"
)

print(
    "RMSE :",
    round(linear_rmse, 2),
    "Billion USD"
)

print(
    "MAPE :",
    round(linear_mape, 2),
    "%"
)

print(
    "R2   :",
    round(linear_r2, 2)
)


print("\n========== DECISION TREE ==========")

print(
    "MAE  :",
    round(tree_mae, 2),
    "Billion USD"
)

print(
    "RMSE :",
    round(tree_rmse, 2),
    "Billion USD"
)

print(
    "MAPE :",
    round(tree_mape, 2),
    "%"
)

print(
    "R2   :",
    round(tree_r2, 2)
)


print("\n========== RANDOM FOREST ==========")

print(
    "MAE  :",
    round(forest_mae, 2),
    "Billion USD"
)

print(
    "RMSE :",
    round(forest_rmse, 2),
    "Billion USD"
)

print(
    "MAPE :",
    round(forest_mape, 2),
    "%"
)

print(
    "R2   :",
    round(forest_r2, 2)
)


# ============================================================
# 6. GRAPHS / VISUALIZATIONS
# ============================================================

print("\n============================================================")
print("              6. GRAPHS / VISUALIZATIONS")
print("============================================================")


# ------------------------------------------------------------
# Graph 1: India GDP Trend
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    df.index,
    df["GDP_Trillion_USD"],
    marker="o",
    markersize=3,
    linewidth=1.5
)

plt.xlabel("Year")
plt.ylabel("GDP (Trillion US$)")
plt.title("India GDP Trend")

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ------------------------------------------------------------
# Graph 2: Training and Testing
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    train.index,
    train["GDP"] / 1e12,
    label="Training Data"
)

plt.plot(
    test.index,
    test["GDP"] / 1e12,
    marker="o",
    label="Actual Test Data"
)

plt.plot(
    test.index,
    linear_pred / 1e12,
    marker="o",
    linestyle="--",
    label="Linear Regression"
)

plt.xlabel("Year")
plt.ylabel("GDP (Trillion US$)")
plt.title("Training and Testing of India GDP")

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ------------------------------------------------------------
# Graph 3: Actual vs All Predictions
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    result["Year"],
    result["Actual GDP (Billion USD)"],
    marker="o",
    linewidth=2,
    label="Actual GDP"
)

plt.plot(
    result["Year"],
    result["Linear Regression"],
    marker="o",
    linestyle="--",
    label="Linear Regression"
)

plt.plot(
    result["Year"],
    result["Decision Tree"],
    marker="o",
    linestyle="--",
    label="Decision Tree"
)

plt.plot(
    result["Year"],
    result["Random Forest"],
    marker="o",
    linestyle="--",
    label="Random Forest"
)

plt.xlabel("Year")
plt.ylabel("GDP (Billion US$)")
plt.title("Actual vs Predicted India GDP")

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ------------------------------------------------------------
# Graph 4: Year-on-Year GDP Growth
# ------------------------------------------------------------

growth_data = df.dropna(
    subset=["Growth_Percentage"]
)


plt.figure(figsize=(10, 5))

plt.bar(
    growth_data.index,
    growth_data["Growth_Percentage"]
)

plt.xlabel("Year")
plt.ylabel("GDP Growth (%)")
plt.title("Year-on-Year GDP Growth of India")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


# ------------------------------------------------------------
# Graph 5: MAE Comparison
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    metrics["Algorithm"],
    metrics["MAE (Billion USD)"]
)

plt.xlabel("Algorithm")
plt.ylabel("MAE (Billion USD)")
plt.title("MAE Comparison")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


# ------------------------------------------------------------
# Graph 6: RMSE Comparison
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    metrics["Algorithm"],
    metrics["RMSE (Billion USD)"]
)

plt.xlabel("Algorithm")
plt.ylabel("RMSE (Billion USD)")
plt.title("RMSE Comparison")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


# ------------------------------------------------------------
# Graph 7: MAPE Comparison
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    metrics["Algorithm"],
    metrics["MAPE (%)"]
)

plt.xlabel("Algorithm")
plt.ylabel("MAPE (%)")
plt.title("MAPE Comparison")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


# ------------------------------------------------------------
# Graph 8: R2 Score Comparison
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    metrics["Algorithm"],
    metrics["R2 Score"]
)

plt.xlabel("Algorithm")
plt.ylabel("R2 Score")
plt.title("R2 Score Comparison")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n============================================================")
print("                    PROJECT SUMMARY")
print("============================================================")

print(
    "Dataset        : World Bank India GDP"
)

print(
    "Data Period    :",
    int(df.index.min()),
    "-",
    int(df.index.max())
)

print(
    "Training Data  :",
    int(train.index.min()),
    "-",
    int(train.index.max())
)

print(
    "Testing Data   :",
    int(test.index.min()),
    "-",
    int(test.index.max())
)

print(
    "Algorithms     : Linear Regression, "
    "Decision Tree Regression, "
    "Random Forest Regression"
)

print("\n========== FINAL MODEL COMPARISON ==========")

print(
    metrics.round(2).to_string(index=False)
)

print("\nProject execution completed successfully.")