from flask import Flask, request, render_template
from model.model import TextClassifier

application = Flask(__name__)

# Load the classifier when the app starts
classifier = TextClassifier(
    './model/trained_model_lr.pkl',
    './model/vectorizer_lr.pkl',
    './model/trained_model_xgb.pkl',
    './model/vectorizer_xgb.pkl'
)

@application.route('/', methods=['GET', 'POST'])
def get_prediction():
    prediction = None
    
    if request.method == 'POST':
        text = request.form.get("text")

        if not text:
            return "No text provided", 400

        prediction = classifier.predict(text)

    return render_template("index.html", prediction=prediction)

if __name__ == '__main__':
    print("Starting the Flask application...")
    application.run(debug=True, host='0.0.0.0',port=8000)