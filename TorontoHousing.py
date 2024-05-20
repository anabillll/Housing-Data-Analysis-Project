import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

# Read the CSV file with the appropriate encoding
file_path = 'E:/TorontoHousing/HousingData.csv'
df = pd.read_csv(file_path, encoding='latin1')

# Print column names
print("Column names in the dataset:")
print(df.columns)

# Display basic statistics
print(df.describe())

# Plot heatmap of correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')

# Show heatmap
plt.show()

# Plot histogram of price
plt.figure(figsize=(8, 6))
sns.histplot(df['Price'], kde=True, color='blue', bins=30)
plt.title('Histogram of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')

# Show histogram
plt.show()

# Plot pairplot
sns.pairplot(df)
plt.suptitle('Pairplot of Housing Data', y=1.02)

# Show pairplot
plt.show()

# Plot bar plot of city vs price
plt.figure(figsize=(10, 6))
sns.barplot(x='City', y='Price', data=df, ci=None)
plt.title('Bar Plot of City vs Price')
plt.xlabel('City')
plt.ylabel('Price')

# Show bar plot
plt.show()

# Plot density plot of price
plt.figure(figsize=(8, 6))
sns.kdeplot(df['Price'], fill=True, color='green')
plt.title('Density Plot of Price')
plt.xlabel('Price')
plt.ylabel('Density')

# Show density plot
plt.show()

# Linear regression on price vs number_beds
X = df[['Number_Beds']]
y = df['Price']
model = LinearRegression()
model.fit(X, y)

# Plot the regression line
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title('Linear Regression: Price vs Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price')

# Show the plot
plt.show()
