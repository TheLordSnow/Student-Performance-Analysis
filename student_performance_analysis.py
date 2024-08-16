
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Setting up the visualization style
sns.set(style="whitegrid")

# Optional: Display all columns of the DataFrame
pd.set_option('display.max_columns', None)

# Load the dataset
# Replace 'student_performance.csv' with your dataset path
df = pd.read_csv('student_performance.csv')

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Handling missing values (if any)
# Example: Filling missing numerical values with the median
df.fillna(df.median(), inplace=True)

# Handling categorical missing values
df['Parental_Education'].fillna(df['Parental_Education'].mode()[0], inplace=True)

# Drop duplicates if any
df.drop_duplicates(inplace=True)

# Verify the cleaning
print(df.isnull().sum())
print(df.info())

# Descriptive statistics
print(df.describe())

# Visualizing distributions of test scores
plt.figure(figsize=(10, 6))
sns.histplot(df['Math_Score'], kde=True, color='blue', label='Math')
sns.histplot(df['Reading_Score'], kde=True, color='green', label='Reading')
sns.histplot(df['Writing_Score'], kde=True, color='red', label='Writing')
plt.legend()
plt.title('Distribution of Test Scores')
plt.show()

# Analyzing correlations
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Boxplot of test scores by parental education level
plt.figure(figsize=(10, 6))
sns.boxplot(x='Parental_Education', y='Math_Score', data=df)
plt.title('Math Scores by Parental Education Level')
plt.xticks(rotation=45)
plt.show()

# Gender vs. Performance
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Math_Score', data=df)
plt.title('Math Scores by Gender')
plt.show()

# T-test to compare scores between genders
male_scores = df[df['Gender'] == 'Male']['Math_Score']
female_scores = df[df['Gender'] == 'Female']['Math_Score']

t_stat, p_val = stats.ttest_ind(male_scores, female_scores)
print(f"T-test results: t-statistic = {t_stat}, p-value = {p_val}")

# Regression Analysis: Impact of study time on math scores
import statsmodels.api as sm

# Independent variable (Study Time)
X = df['Study_Time']
# Dependent variable (Math Score)
y = df['Math_Score']

# Add constant to the model (intercept)
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())

# Save the cleaned data for future use
df.to_csv('cleaned_student_performance.csv', index=False)
