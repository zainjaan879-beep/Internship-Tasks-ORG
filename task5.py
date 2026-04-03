import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

sns.set_style("whitegrid")

# 1. Load Dataset
df = pd.read_csv('bank-additional-full.csv', sep=';')  # Separator semicolon

# 2. Dataset Understanding
print("Shape of dataset:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
df.info()
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Basic EDA
plt.figure(figsize=(6,4))
sns.countplot(x='y', data=df)
plt.title("Distribution of Personal Loan Acceptance (y)")
plt.show()

# 4. Model Training (Logistic Regression)
X = df[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan']]  # basic features
y = df['y']

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)

# 6. Conclusion
print("""
Conclusion:
1 Dataset loaded correctly with semicolon separator.
2 EDA shows distribution of customers who accepted the offer.
3 Logistic Regression model trained on basic customer features.
4 Accuracy and confusion matrix give model performance.
""")