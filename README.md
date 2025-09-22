# Task 1 - Data Cleaning & Preprocessing

I built a preprocessing pipeline for the Titanic dataset. No dataset is included here — you can plug in your own CSV.

## What I did
- Checked null values and data types
- Filled missing values (Age, Fare, Embarked) and dropped Cabin
- Encoded categorical features (Sex, Embarked)
- Standardized numeric features
- Removed outliers using IQR for Age and Fare
- Saved cleaned dataset and boxplots

## How to run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
