### The text classification ml model deployed with Flask app and Docker.


#### model.py

ML Python code for text classification.
When file is running the model is being fittted.
Model saved to model.h5 file.

#### app.py

The Flask app file with predict function.
When running the index.html page is created. The prediction could be checked in browser.

#### Emotion_final.csv

The data: column 'Text' (string sentences) and target column 'Emotion' (6 target moods)   
labels = ['anger', 'fear', 'happy', 'love', 'sadness', 'surprise']
dataset from kaggle

#### requirements.txt

required libraries.

#### templates (folder):
index.html (web page)

#### model.h5

model created during the model.py run.
