# train_model.py

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)

# Save model to file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")
