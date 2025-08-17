# Step 1: Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load the dataset
# Assuming you downloaded "StudentsPerformance.csv" from Kaggle
df = pd.read_csv("StudentsPerformance.csv")

# Step 3: Display first 5 rows
print("First 5 rows of dataset:")
print(df.head())

# Step 4: Check shape (rows, columns)
print("\nShape of dataset:", df.shape)

# Step 5: Info about dataset
print("\nDataset Info:")
print(df.info())

# Step 6: Check missing values
print("\nMissing values:")
print(df.isnull().sum())






# ----- Block A: CLEAN + FEATURE ENGINEERING -----

# Keeping a raw copy
df_raw = df.copy()

# Standardize/shorten column names
df = df.rename(columns={
    'math score': 'math_score',
    'reading score': 'reading_score',
    'writing score': 'writing_score',
    'parental level of education': 'parental_education',
    'race/ethnicity': 'race_ethnicity',
    'test preparation course': 'test_prep'
})

# Making sure score columns are numeric:
score_cols = ['math_score', 'reading_score', 'writing_score']
df[score_cols] = df[score_cols].apply(pd.to_numeric, errors='coerce')

# Creating helpful features:
df['average'] = df[score_cols].mean(axis=1)
df['total']   = df[score_cols].sum(axis=1)

# Pass/Fail rule:
df['result'] = np.where(df['average'] >= 33, 'Pass', 'Fail')

# Letter grade buckets (A/B/C/D) - optional but nice for analysis
def to_grade(avg):
    if avg >= 90: return 'A'
    if avg >= 80: return 'B'
    if avg >= 60: return 'C'
    if avg >= 45: return 'D'
    if avg >= 33: return 'E'
    return 'F'

# --- Add this right after `to_grade` is defined ---
def add_engineered_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    score_cols_local = ['math_score', 'reading_score', 'writing_score']
    df_out['average'] = df_out[score_cols_local].mean(axis=1)
    df_out['total']   = df_out[score_cols_local].sum(axis=1)
    df_out['grade']   = df_out['average'].apply(to_grade)
    return df_out


df['grade'] = df['average'].apply(to_grade)

# Mark categorical columns explicitly (saves memory, helps later)
cat_cols = ['gender', 'race_ethnicity', 'parental_education', 'lunch', 'test_prep', 'result', 'grade']
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype('category')

print("Columns now:", df.columns.tolist())

# Sample of new structured dataset.
print("\nSample after cleaning & features:\n", df.head(3))







# BLOCK B: Quick EDA
print("\n--- Dataset Description ---")
print(df.describe(include='all'))

print("\n--- Pass/Fail Distribution ---")
print(df['result'].value_counts())

print("\n--- Grade Distribution ---")
print(df['grade'].value_counts())


# NOTE:
# 1. Use observed=False if you want to keep all possible categories.
# 2. Use observed=True if you only care about the categories that appear in your data (usually preferred in analysis).


print("\n--- Group by Gender (Average Scores) ---")
print(df.groupby('gender', observed=False)[['math_score','reading_score','writing_score','average']].mean())

print("\n--- Group by Test Preparation Course ---")
print(df.groupby('test_prep', observed=False)[['average']].mean())

print("\n--- Group by Lunch Type ---")
print(df.groupby('lunch', observed=False)[['average']].mean())







# Block C: Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# 1. matplotlib.pyplot (plt): A plotting library that allows us to create graphs, charts, and plots in Python.

# 2. seaborn (sns): Built on top of matplotlib; it makes statistical plots more visually appealing and easier to create.


plt.figure(figsize=(8,6))

# This is used to encode the categorical columns in dataset into numeric(float). This is done so that Correlation Heatmap is drawn properly. 
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include=['object', 'category']):
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()

# 🔹 Line 1: plt.figure(figsize=(8,6))

# This creates a new figure (canvas) of size 8x6 inches where our plot will be drawn.

# figsize controls how big the chart will look.

# 🔹 Line 2: sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

# df.corr() → Calculates the correlation matrix of all numeric columns.

# Correlation measures how strongly two variables are related (values range from -1 to +1).

# Example: If Study Hours ↑ and Exam Score ↑ together, correlation is close to +1.

# If Absences ↑ and Exam Score ↓, correlation is negative.

# sns.heatmap(..., annot=True, cmap="coolwarm") → Creates a heatmap (colored grid) of correlations.

# annot=True → Displays actual correlation values inside each box.

# cmap="coolwarm" → Uses a blue-to-red color scale (blue = negative correlation, red = positive).

# 🔹 Line 3: plt.title("Correlation Heatmap")

# Adds a title to the plot so the viewer knows what the chart represents.

# 🔹 Line 4: plt.show()

# Finally, this displays the chart on the screen.

# Without plt.show(), the plot may not appear in some environments.









# 🔹 Block D – Splitting the Data & Model Training

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Step 1: Split into features & target
# Define features (all columns except 'Outcome') and target (the column to predict)
X = df.drop("result", axis=1)   # Features
y = df["result"]                # Target variable

# Step 2: Convert categorical columns into numeric
X = pd.get_dummies(X, drop_first=True)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)

feature_columns = X.columns.tolist()








# ---------------- BLOCK E : Model Evaluation ----------------

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# 3. Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 4. Visualize Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()







# ---------------- Block F: Prediction on new data ----------------

# Sample new student data (change values as per requirement)
new_data = pd.DataFrame({
    'gender': ['female'],
    'race_ethnicity': ['group B'],
    'parental_education': ["bachelor's degree"],
    'lunch': ['standard'],
    'test_prep': ['completed'],
    'math_score': [85],
    'reading_score': [90],
    'writing_score': [88]
})

# ✅ Add engineered features FIRST (average, total, grade)
new_data = add_engineered_features(new_data)

# Apply the same preprocessing as training data
new_data_encoded = pd.get_dummies(new_data, drop_first=True)

# Align with training data columns (to avoid column mismatch error)
new_data_encoded = new_data_encoded.reindex(columns=X_train.columns, fill_value=0)

# Predict outcome
prediction = model.predict(new_data_encoded)
prediction_proba = model.predict_proba(new_data_encoded)

# Show results
print("Prediction (Outcome):", prediction[0])
print("Prediction Probabilities (Fail/Pass):", prediction_proba[0])






# ----------------- Block G2: Model Evaluation -----------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# Make predictions on the test set
y_pred = model.predict(X_test)

# 1. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")

# 2. Precision (macro handles multi-class well)
precision = precision_score(y_test, y_pred, average='macro')
print(f"✅ Precision (macro): {precision:.4f}")

# 3. Recall
recall = recall_score(y_test, y_pred, average='macro')
print(f"✅ Recall (macro): {recall:.4f}")

# 4. F1-Score
f1 = f1_score(y_test, y_pred, average='macro')
print(f"✅ F1-Score (macro): {f1:.4f}")

# 5. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n📊 Confusion Matrix:")
print(cm)

# 6. Classification Report
print("\n📑 Classification Report:")
print(classification_report(y_test, y_pred))

artifacts = {"model": model, "feature_columns": feature_columns}
joblib.dump(artifacts, "student_performance_model.pkl")
print("💾 Model+columns saved as student_performance_model.pkl")








# ----------------- Block G3: Interactive User Input System (CLI) -----------------
import joblib

art = joblib.load("student_performance_model.pkl")
loaded_model = art["model"]
feature_columns = art["feature_columns"]
print("📂 Model loaded successfully!")

print("\n🎓 Welcome to the Student Performance Prediction System 🎓")

# Collect user inputs
gender = input("Enter Gender (male/female): ").strip().lower()
race_ethnicity = input("Enter Race/Ethnicity (group A/B/C/D/E): ").strip()
parental_education = input("Enter Parental Education (e.g., bachelor's degree, some college, etc.): ").strip().lower()
lunch = input("Enter Lunch Type (standard/free/reduced): ").strip().lower()
test_prep = input("Enter Test Preparation Course (completed/none): ").strip().lower()
math_score = float(input("Enter Math Score (0-100): "))
reading_score = float(input("Enter Reading Score (0-100): "))
writing_score = float(input("Enter Writing Score (0-100): "))

# Put input into a DataFrame
user_data = pd.DataFrame({
    'gender': [gender],
    'race_ethnicity': [race_ethnicity],
    'parental_education': [parental_education],
    'lunch': [lunch],
    'test_prep': [test_prep],
    'math_score': [math_score],
    'reading_score': [reading_score],
    'writing_score': [writing_score]
})

# ✅ Add engineered features
user_data = add_engineered_features(user_data)

# Apply same preprocessing as training
user_data_encoded = pd.get_dummies(user_data, drop_first=True)
user_data_encoded = user_data_encoded.reindex(columns=feature_columns, fill_value=0)

# Make prediction
user_prediction = loaded_model.predict(user_data_encoded)[0]
user_prediction_proba = loaded_model.predict_proba(user_data_encoded)[0]


# Show results
print("\n🔮 Prediction Result:")
print(f"Student is likely to: {user_prediction}")
print(f"Prediction Probabilities (Fail/Pass): {user_prediction_proba}")
