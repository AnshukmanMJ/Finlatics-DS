# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
np.random.seed(42)
num_customers = 500
data = pd.DataFrame({
    'CustomerID': range(1, num_customers + 1),
    'Age': np.random.randint(18, 65, size=num_customers),
    'Gender': np.random.choice(['Male', 'Female'], size=num_customers),
    'TenureMonths': np.random.randint(1, 60, size=num_customers),
    'WebsiteVisits': np.random.randint(1, 30, size=num_customers),
    'EmailsOpened': np.random.randint(0, 50, size=num_customers),
    'AdsClicked': np.random.randint(0, 20, size=num_customers),
    'LastPurchaseDaysAgo': np.random.randint(1, 180, size=num_customers),
    'TotalSpending': np.random.uniform(50, 2000, size=num_customers) * (np.random.rand(num_customers) + 0.5),
    'Churn': np.random.choice(['Yes', 'No'], size=num_customers, p=[0.2, 0.8])
})
print("--- Sample Data ---")
print(data.head())
print("\n--- Data Info ---")
data.info()

print("\n--- 1. Customer Segmentation ---")

features_for_clustering = data[['TenureMonths', 'WebsiteVisits', 'TotalSpending']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_for_clustering)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show()

optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
data['CustomerSegment'] = kmeans.fit_predict(scaled_features)

segment_analysis = data.groupby('CustomerSegment')[['TenureMonths', 'WebsiteVisits', 'TotalSpending']].mean().reset_index()
print("\n--- Analysis of Customer Segments ---")
print(segment_analysis)

plt.figure(figsize=(12, 6))
sns.scatterplot(x='TenureMonths', y='TotalSpending', hue='CustomerSegment', data=data, palette='viridis', s=100, alpha=0.7)
plt.title('Customer Segments by Tenure and Total Spending')
plt.xlabel('Tenure (Months)')
plt.ylabel('Total Spending')
plt.legend(title='Customer Segment')
plt.show()

print("\n--- 2. Churn Prediction ---")
data_ml = pd.get_dummies(data, columns=['Gender'], drop_first=True)
data_ml['Churn'] = data_ml['Churn'].map({'Yes': 1, 'No': 0})

X = data_ml.drop(['CustomerID', 'Churn'], axis=1)
y = data_ml['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
plt.title('Confusion Matrix for Churn Prediction')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

churn_by_segment = data.groupby('CustomerSegment')['Churn'].value_counts(normalize=True).unstack().fillna(0)
print("\n--- Churn Rate by Customer Segment ---")
print(churn_by_segment)

churn_by_segment.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
plt.title('Churn Rate by Customer Segment')
plt.xlabel('Customer Segment')
plt.ylabel('Proportion of Customers')
plt.xticks(rotation=0)
plt.legend(title='Churn')
plt.show()