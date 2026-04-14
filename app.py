from flask import Flask, render_template, request
import numpy as np
import pickle
import os
app = Flask(__name__)

# Define the NamedModel class so pickle can deserialize it
class NamedModel:
    def __init__(self, clf, names):
        self.clf = clf
        self.names = names
    def predict(self, X):
        return [self.names[i] for i in self.clf.predict(X)]
    def predict_proba(self, X):
        return self.clf.predict_proba(X)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

species_info = {
    'setosa': {
        'image': 'setosa.jpg',
        'description': 'Iris Setosa is a small, hardy species with distinctly smaller petals. It is the most easily distinguishable of the three species.',
        'color': '#e74c8b'
    },
    'versicolor': {
        'image': 'versicolor.jpg',
        'description': 'Iris Versicolor, also known as the Blue Flag Iris, features medium-sized flowers with beautiful violet-blue petals.',
        'color': '#7c3aed'
    },
    'virginica': {
        'image': 'virginica.jpg',
        'description': 'Iris Virginica is the largest of the three species, featuring large, showy flowers with broad petals, typically purple or violet.',
        'color': '#0ea5e9'
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width  = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width  = float(request.form['petal_width'])

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        confidence = round(max(proba) * 100, 2)

        species_name = prediction.lower()
        info = species_info.get(species_name, {
            'image': 'default.jpg',
            'description': 'A beautiful Iris species.',
            'color': '#10b981'
        })

        return render_template(
            'result.html',
            prediction=prediction.capitalize(),
            confidence=confidence,
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
            image=info['image'],
            description=info['description'],
            accent_color=info['color']
        )
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
