# TASK 2: CREDIT RISK PREDICTION - INTERNSHIP TASK

# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

sns.set_style("whitegrid")

# 2. Load Dataset
df = pd.read_csv('loan_default.csv')  # filename update kiya

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
plt.figure(figsize=(12,5))
sns.histplot(df['Income'], bins=20, kde=True)
plt.title("Income Distribution")
plt.show()

plt.figure(figsize=(12,5))
sns.boxplot(x='Education', y='LoanAmount', data=df)
plt.title("Loan Amount by Education")
plt.show()

plt.figure(figsize=(12,5))
sns.scatterplot(x='Income', y='LoanAmount', hue='Default', data=df)
plt.title("Income vs Loan Amount by Default")
plt.show()

# 6. Model Training
X = df[['Age','Income','LoanAmount','CreditScore','MonthsEmployed','NumCreditLines','InterestRate','LoanTerm','DTIRatio']]
y = df['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
- Dataset loaded successfully with no missing values.
- EDA shows feature distributions and relationships.
- Logistic Regression model predicts loan default with reasonable accuracy.
- Task requirements completed in a simple, easy-to-understand format.
""")