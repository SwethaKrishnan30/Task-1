import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the CSV file
file_path = r'C:\Users\Joshua\Downloads\Iris.csv'
df = pd.read_csv(file_path)

# Drop the 'Id' column as it's not needed for classification
df.drop(columns=['Id'], inplace=True)

# Define features and target
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=1)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

# Visualize the results
plt.figure(figsize=(10, 6))

# Plot actual classes
sns.scatterplot(x=X_test['SepalLengthCm'], y=X_test['SepalWidthCm'], hue=y_test, palette='Set1', marker='o', s=100, label='Actual')

# Plot predicted classes
sns.scatterplot(x=X_test['SepalLengthCm'], y=X_test['SepalWidthCm'], hue=y_pred, palette='Set2', marker='x', s=100, label='Predicted')

plt.title('KNN Classification Results on Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.show()
