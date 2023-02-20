import numpy as np
import pandas as pd
from flask import Flask, request
from flask import jsonify, render_template
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Create flask app
flask_app = Flask(__name__)
model = load_model("model.h5")

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route('/predict', methods=['POST'])
def predict():
    # Get the text input from the request
    text = request.form['text']

    df = pd.read_csv("Emotion_final.csv")
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['Text'])

    # Preprocess the text input
    text = [text]
    text = tokenizer.texts_to_sequences(text)
    text = pad_sequences(text, maxlen=100)

    # Make a prediction with the model
    prediction = model.predict(text)

    # Get the label with the highest probability
    predicted_label = np.argmax(prediction)

    # Convert the label back to an emotion string
    labels = ['anger', 'fear', 'happy', 'love', 'sadness', 'surprise']
    predicted_emotion = labels[predicted_label]

    # Return the predicted emotion
    return jsonify({'emotion': predicted_emotion})

if __name__ == "__main__":
    flask_app.run(debug=True)
