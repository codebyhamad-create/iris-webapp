from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pickle

# Define the NamedModel class to match the logic in app.py
# This allows the model to return species names directly
class NamedModel:
    def __init__(self, clf, names):
        self.clf = clf
        self.names = names
    def predict(self, X):
        return [self.names[i] for i in self.clf.predict(X)]
    def predict_proba(self, X):
        return self.clf.predict_proba(X)

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
cv_score = cross_val_score(model, X, y, cv=5).mean()

print("Test Accuracy:", accuracy)
print("Cross Validation Score:", cv_score)

# Wrap the model and save it to the correct path for app.py
named_model = NamedModel(model, data.target_names)
with open("model.pkl", "wb") as f:
    pickle.dump(named_model, f)