import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'StudyHours': [2, 4, 6, 1, 5, 3, 7, 2, 8, 4],
    'PreviousGrade': [50, 60, 80, 45, 75, 55, 85, 48, 90, 65],
    'AttendanceRate': [60, 70, 90, 50, 85, 65, 95, 55, 98, 75],
    'Participation': [1, 2, 3, 1, 3, 2, 3, 1, 3, 2],
    'Pass': [0, 1, 1, 0, 1, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['StudyHours', 'PreviousGrade', 'AttendanceRate', 'Participation']]
y = df['Pass']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Predict for new student input (with proper column names to avoid warning)
new_input = pd.DataFrame(np.array([[8, 90, 98, 3]]),
                         columns=['StudyHours', 'PreviousGrade', 'AttendanceRate', 'Participation'])

prediction = model.predict(new_input)
print(f"Predicted Output (1=Pass, 0=Fail): {prediction[0]}")
