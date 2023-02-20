import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load the csv file
df = pd.read_csv("Emotion_final.csv")

print(df.head())

# Convert emotions to numeric labels
labels = df['Emotion'].unique()
value_to_number = {value: number for number, value in enumerate(np.unique(labels))}
df['Emotion'] = df['Emotion'].replace(value_to_number)
y = df['Emotion']

# Preprocess the text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['Text'])
X = tokenizer.texts_to_sequences(df['Text'])
X = pad_sequences(X, maxlen=100)

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15, random_state=42, stratify=y)

# Build the model
model = Sequential()
model.add(Embedding(5000, 128, input_length=X.shape[1]))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(6, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Make h5 file of our model
model.save("model.h5")

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

# Analyze the accuracy
print('Test accuracy:', accuracy)
if accuracy > 0.85:
    print('The model is good!')
elif accuracy > 0.7:
    print('The model is okay, but could be improved.')
else:
    print('The model needs improvement!')

# Analyze underfitting/overfitting
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=False)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=False)
print('Train accuracy:', train_accuracy)
print('Test accuracy:', test_accuracy)
if train_accuracy - test_accuracy > 0.1:
    print('The model is overfitting.')
elif test_accuracy - train_accuracy > 0.1:
    print('The model is underfitting.')
else:
    print('The model is good!')
