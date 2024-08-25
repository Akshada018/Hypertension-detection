# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv('data/hypertension_data.csv')

# Data preprocessing
data.dropna(inplace=True)
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Feature selection
features = ['Age', 'BMI', 'Cholesterol', 'BloodSugar', 'Gender']
X = data[features]
y = data['Hypertension']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and the scaler
with open('model/hypertension_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('model/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
