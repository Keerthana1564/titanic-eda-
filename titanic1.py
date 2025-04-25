# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Step 2: Load the dataset
df = pd.read_csv('titanic.csv')  # Make sure 'train.csv' is uploaded or present in your folder
print("First 5 rows of data:")
print(df.head())

# Step 3: Understand the data
print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Step 4: Visualization

# 4.1 Histograms
print("\nPlotting Histograms...")
df.hist(figsize=(12,10))
plt.suptitle('Histograms of Titanic Dataset')
plt.show()

# 4.2 Boxplots
print("\nPlotting Boxplot of Age vs Survived...")
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Boxplot of Age vs Survived')
plt.show()

# 4.3 Correlation Heatmap
print("\nPlotting Correlation Heatmap...")
plt.figure(figsize=(10,8))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 4.4 Pairplot
print("\nPlotting Pairplot...")
sns.pairplot(df)
plt.show()

# Step 5: Analysis and Insights

# 5.1 Survival Rate by Gender
print("\nSurvival Rate by Gender:")
print(df.groupby('Sex')['Survived'].mean())

# 5.2 Survival Rate by Passenger Class
print("\nSurvival Rate by Passenger Class:")
print(df.groupby('Pclass')['Survived'].mean())

# Step 6: Interactive Plot using Plotly
print("\nPlotting Interactive Age Distribution by Survival Status...")
fig = px.histogram(df, x="Age", color="Survived", nbins=30, title="Age vs Survived (Interactive)")
fig.show()