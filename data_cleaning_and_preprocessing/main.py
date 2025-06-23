import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load Dataset
df = pd.read_csv('Titanic-Dataset.csv')
print("Original Shape:", df.shape)
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

# Handle Missing Values
df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

# Fill Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode Categorical Variables
label_enc = LabelEncoder()
df['Sex'] = label_enc.fit_transform(df['Sex'])        # Male:1, Female:0
df['Embarked'] = label_enc.fit_transform(df['Embarked'])

# Normalize Numerical Features
scaler = StandardScaler()
num_cols = ['Age', 'Fare']
df[num_cols] = scaler.fit_transform(df[num_cols])

# Visualize Outliers with Boxplots
plt.figure(figsize=(10, 4))
for i, col in enumerate(num_cols):
    plt.subplot(1, 2, i+1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Remove Outliers using IQR
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

print("\nFinal Cleaned Data Shape:", df.shape)
print(df.head())