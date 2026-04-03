# IRIS DATASET ANALYSIS - INTERNSHIP TASK

# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

sns.set_style("whitegrid")


# 2. Load Dataset
try:
    df = pd.read_csv('iris.csv')
except:
    df = sns.load_dataset('iris')


# 3. Dataset Understanding
print("Shape of dataset:", df.shape)
print("Columns:", df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# 4. Data Cleaning
print("\nMissing Values:")
print(df.isnull().sum())


# 5. Exploratory Data Analysis (EDA)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.scatterplot(
    x='sepal_length', 
    y='sepal_width', 
    hue='species', 
    data=df, 
    ax=axes[0]
)
axes[0].set_title("Sepal Length vs Sepal Width")

sns.histplot(
    df['petal_length'], 
    bins=20, 
    kde=True, 
    ax=axes[1], 
    color='green'
)
axes[1].set_title("Petal Length Distribution")

sns.boxplot(
    x='species', 
    y='sepal_length', 
    data=df, 
    ax=axes[2]
)
axes[2].set_title("Sepal Length by Species")

plt.tight_layout()
plt.show()


# 6. Model Training
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# 7. Model Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(cm)


# 8. Conclusion
print("""
Conclusion:
- Dataset is clean with no missing values.
- EDA shows feature relationships and distributions.
- Logistic Regression model achieved good accuracy.
- All task requirements are successfully completed.
""")