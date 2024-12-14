import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv('housing.csv')

print("\nDataset Overview:\n")
print(df.info())
print("\nBasic Statistics:\n")
print(df.describe())

missing_values = df.isnull().sum()
print("\nMissing Values in Each Column:\n", missing_values)

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Visualization")
plt.show()

df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['median_house_value'], kde=True, bins=50)
plt.title("Distribution of Median House Value")
plt.xlabel("Median House Value")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['median_income'], kde=True, bins=50)
plt.title("Distribution of Median Income")
plt.xlabel("Median Income")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='ocean_proximity_NEAR BAY', y='median_house_value', data=df)
plt.title("Median House Value by Ocean Proximity")
plt.xlabel("Ocean Proximity")
plt.ylabel("Median House Value")
plt.show()



path ="housing.csv"


def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    df['median_house_value'] = df['median_house_value'] / 1_000_000
    df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)
    numerical_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
