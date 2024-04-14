import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
from keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.sequence import Tokenizer

import joblib
# Load the labeled data
data = pd.read_csv('checkworthiness_labeled.csv')
# Preprocess the data
data['Category'] = data['Category'].map({'Yes': 1, 'No': 0})
# Split the data into training and testing sets
X = data['Text']
y = data['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
# Padding sequences
max_len = max([len(x) for x in X_train_seq])
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')
# Build the model
model = Sequential([
   Embedding(input_dim=5000, output_dim=16, input_length=max_len),
   GlobalAveragePooling1D(),
   Dense(24, activation='relu'),
   Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
model.fit(X_train_pad, y_train, epochs=200, batch_size=32, validation_data=(X_test_pad, y_test))
# Evaluate the model
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')
# Save the trained model
model.save('checkworthiness_model_advanced_tf.keras')
# Load the trained model
model = tf.keras.models.load_model('checkworthiness_model_advanced_tf.keras')
# Read the leaderboard data
leaderboard_data = pd.read_csv('checkworthiness_leaderboard.csv')
# Tokenization and padding for leaderboard data
leaderboard_data_seq = tokenizer.texts_to_sequences(leaderboard_data['Text'])
leaderboard_data_pad = pad_sequences(leaderboard_data_seq, maxlen=max_len, padding='post', truncating='post')
# Apply the model to each row and update the Category field
leaderboard_data['Category'] = (model.predict(leaderboard_data_pad) > 0.5).astype(int).flatten()
leaderboard_data['Category'] = leaderboard_data['Category'].map({1: 'Yes', 0: 'No'})
# Save the updated leaderboard data
leaderboard_data.to_csv('checkworthiness_leaderboard_updated_advanced_tf.csv', index=False)
print("Advanced TensorFlow model updated leaderboard saved successfully!")