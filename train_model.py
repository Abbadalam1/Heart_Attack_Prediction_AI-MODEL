import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load UCI Heart Disease dataset
# CHANGE HERE: Ensure 'heart.csv' is in the same directory as this script
data = pd.read_csv('heart.csv')

# Binarize the target column ('num'): 0 = no heart disease, 1-4 = heart disease (convert to 1)
data['num'] = data['num'].apply(lambda x: 1 if x > 0 else 0)

# Split into features and target
# CHANGE HERE: Use 'num' instead of 'target' as the target column
X = data.drop('num', axis=1)
y = data['num']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Save model
# CHANGE HERE: Ensure the output path is writable; 'heart_disease_model.pkl' will be created
joblib.dump(model, 'heart_disease_model.pkl')