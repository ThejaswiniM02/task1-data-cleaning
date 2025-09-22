# preprocess.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_dataset():
    # Direct download from public URL (no KaggleHub needed)
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    print("Dataset loaded successfully from URL")
    return df

def preprocess(df):
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop('Cabin', axis=1, inplace=True)

    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    # Standardize numerical features
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

    # Remove outliers using IQR for Fare
    Q1 = df['Fare'].quantile(0.25)
    Q3 = df['Fare'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['Fare'] >= Q1 - 1.5*IQR) & (df['Fare'] <= Q3 + 1.5*IQR)]
    return df

def main(outdir="output"):
    df = load_dataset()
    df_cleaned = preprocess(df)

    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "titanic_cleaned.csv")
    df_cleaned.to_csv(outpath, index=False)
    print(f"Cleaned dataset saved to {outpath}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="output", help="Output directory")
    args = parser.parse_args()
    main(args.outdir)

