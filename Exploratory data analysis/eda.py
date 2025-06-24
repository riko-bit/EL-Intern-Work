import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set(style="whitegrid")

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Drop irrelevant columns
df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode categorical features
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# ==================== Summary Statistics ====================
print("Summary Statistics:\n")
print(df.describe(include='all'))

# ==================== Histograms ====================
print("\nGenerating histograms...")
df.hist(figsize=(10, 8), bins=20, edgecolor='black', grid=False)
plt.suptitle("Histograms of Numeric Features", fontsize=16)
for ax in plt.gcf().axes:
    ax.set_xlabel(ax.get_title())
    ax.set_ylabel("Frequency")
    ax.set_title("")
plt.tight_layout()
plt.savefig("histograms.png", bbox_inches='tight')
plt.show()

# ==================== Boxplots ====================
print("Generating boxplots...")
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplot of Age and Fare", fontsize=14)
plt.ylabel("Value (Original Scale)")
plt.xlabel("Features")
plt.savefig("boxplot_output.png", bbox_inches='tight')
plt.show()

# ==================== Pairplot ====================
print("Generating pairplot...")
sns.pairplot(df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']], hue='Survived', plot_kws={'alpha': 0.7})
plt.suptitle("Pairplot of Key Features by Survival", y=1.02, fontsize=14)
plt.savefig("pairplot.png", bbox_inches='tight')
plt.show()

# ==================== Correlation Matrix ====================
print("Generating correlation heatmap...")
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title("Correlation Matrix of Titanic Features", fontsize=14)
plt.savefig("correlation_matrix.png", bbox_inches='tight')
plt.show()
