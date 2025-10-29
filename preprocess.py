#!/usr/bin/env python3
"""
Preprocess Heart Disease dataset:
- download CSV
- basic cleaning
- train/test split (stratify by target)
- fit OneHotEncoder on train categorical cols and StandardScaler on train numeric cols
- transform train & test and save data/train_encoded.csv and data/test_encoded.csv
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

# ---- CONFIG ----
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Heart%20Disease.csv"

# ----------------

os.makedirs("data", exist_ok=True)

print("Downloading dataset...")
df = pd.read_csv("data/heart.csv")
print(f"Loaded {len(df)} rows — columns: {list(df.columns)}")

# --- Find target column ---
# prefer common names, else fallback to last column
possible_targets = ["target", "Target", "heartdisease", "HeartDisease", "num", "class"]
target_col = None
for c in possible_targets:
    if c in df.columns:
        target_col = c
        break
if target_col is None:
    target_col = df.columns[-1]  # fallback
print(f"Using target column: '{target_col}'")

# basic cleanup
df = df.dropna().drop_duplicates().reset_index(drop=True)
print(f"After dropna/drop_duplicates: {len(df)} rows")

# features / target
X = df.drop(columns=[target_col])
y = df[target_col]

# identify categorical and numeric columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
print(f"Categorical cols: {cat_cols}")
print(f"Numeric cols: {num_cols}")

# train/test split (stratify by target)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# fit encoder on train categorical (if any)
if len(cat_cols) > 0:
    # use sparse=False for compatibility; if your sklearn warns, it's okay
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])
    cat_feature_names = list(encoder.get_feature_names_out(cat_cols))
    X_train_cat_df = pd.DataFrame(X_train_cat, columns=cat_feature_names, index=X_train.index)
    X_test_cat_df = pd.DataFrame(X_test_cat, columns=cat_feature_names, index=X_test.index)
else:
    # create empty DataFrames with zero columns
    cat_feature_names = []
    X_train_cat_df = pd.DataFrame(index=X_train.index)
    X_test_cat_df = pd.DataFrame(index=X_test.index)

# fit scaler on numeric columns (if any)
if len(num_cols) > 0:
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_test_num = scaler.transform(X_test[num_cols])
    X_train_num_df = pd.DataFrame(X_train_num, columns=num_cols, index=X_train.index)
    X_test_num_df = pd.DataFrame(X_test_num, columns=num_cols, index=X_test.index)
else:
    X_train_num_df = pd.DataFrame(index=X_train.index)
    X_test_num_df = pd.DataFrame(index=X_test.index)

# combine numeric + categorical encoded
X_train_final = pd.concat([X_train_num_df.reset_index(drop=True), X_train_cat_df.reset_index(drop=True)], axis=1)
X_test_final = pd.concat([X_test_num_df.reset_index(drop=True), X_test_cat_df.reset_index(drop=True)], axis=1)

# attach target column
train_df = pd.concat([X_train_final, pd.Series(y_train.reset_index(drop=True), name=target_col)], axis=1)
test_df = pd.concat([X_test_final, pd.Series(y_test.reset_index(drop=True), name=target_col)], axis=1)

# save
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)
df.to_csv("data/data_raw.csv", index=False)

print("Saved:")
print(" - data/data_raw.csv")
print(" - data/train.csv")
print(" - data/test.csv")