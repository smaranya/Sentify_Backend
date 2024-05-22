from flask import Flask, request, jsonify
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from nltk.stem import SnowballStemmer
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import warnings
import os
import pickle
from flask_cors import CORS

from keras.models import load_model, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences

from PIL import Image
warnings.filterwarnings("ignore", category=DeprecationWarning)


app = Flask(__name__)
CORS(app)
train = pd.read_csv('train.csv', dtype=str)
# train.columns = ['label', 'text']
combine = train.copy()

ss = SnowballStemmer("english")


def preprocess_data(data):
    def remove_pattern(text, pattern):
        if pd.isnull(text):  # Check if the text is NaN
            return ""
        elif not isinstance(text, str):  # Check if the text is not a string
            return str(text)
        else:
            r = re.findall(pattern, text)
            for i in r:
                text = re.sub(i, "", text)
            return text
# remove user mentions
    data['Tidy_Tweets'] = np.vectorize(
        remove_pattern)(data['text'], "@[\w]*")

    # remove links
    data['Tidy_Tweets'] = data['Tidy_Tweets'].apply(
        lambda x: re.sub(r'https?://\S+', ' ', x))

    # remove punctuations
    # combine['Tidy_Tweets'] = combine['Tidy_Tweets'].str.replace("[^a-zA-Z#]", "")
    data['Tidy_Tweets'] = data['Tidy_Tweets'].apply(
        lambda x: re.sub(r'[^\w\s]', ' ', x))

    # remove short words
    data['Tidy_Tweets'] = data['Tidy_Tweets'].apply(
        lambda x: ' '.join([w for w in x.split() if len(w) > 1]))

    # tokenize
    # tokenized_text = combine['Tidy_Tweets'].apply(lambda x: x.split())
    tokenized_text = data['Tidy_Tweets'].apply(lambda x: nltk.word_tokenize(x))

    tokenized_text = tokenized_text.apply(lambda x: nltk.pos_tag(x))

    tokenized_text = tokenized_text.apply(lambda line: [(word, tag) for (word, tag) in line if tag not in (
        'IN', 'NNP', 'NNPS', 'TO', 'UH', 'CC', 'MD', 'PRP', 'PRP$', 'DT', 'WDT', 'WP', 'WP$', 'WRB')])

    tokenized_text = tokenized_text.apply(
        lambda line: [word for (word, _) in line])

    # stemming
    tokenized_text = tokenized_text.apply(lambda x: [ss.stem(i) for i in x])

    for i in range(len(tokenized_text)):
        tokenized_text[i] = ' '.join(tokenized_text[i])

    data['Tidy_Tweets'] = tokenized_text


preprocess_data(combine)

# Bag-of-Words
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X_train = vectorizer.fit_transform(combine['Tidy_Tweets'])
y_train = train['sentiment']

# Train a Decision Tree classifier
classifier = GradientBoostingClassifier()
classifier.fit(X_train, y_train)

# API endpoint for processing requests


@app.route('/predict_csv', methods=['GET'])
def predict_sentiment():
    # Predict labels for the test set
    test = pd.read_csv('test.csv', dtype=str)

    # Preprocess test data
    preprocess_data(test)

    X_test = vectorizer.transform(test['Tidy_Tweets'])
    predicted_labels = classifier.predict(X_test)

    # test_reset_index = test.reset_index(drop=True)

    # # Add predicted labels to the combined dataset
    # combine.loc[train.shape[0]:, 'predicted_label'] = predicted_labels
    test['predicted_label'] = predicted_labels

    # print(combine[['Tidy_Tweets', 'predicted_label']].tail())
    # Assuming combine dataframe contains the predicted labels in 'predicted_label' column for the test set
    # Count the number of positive and negative comments
    positive_count = (test['predicted_label'] == 'positive').sum()
    negative_count = (test['predicted_label'] == 'negative').sum()
    total_comments = positive_count + negative_count
    positive_percentage = round((positive_count / total_comments) * 100)
    negative_percentage = round((negative_count / total_comments) * 100)

    positive_comments = test.loc[test['predicted_label']
                                 == 'positive', 'text'].head(5).tolist()

    # Filter out the first five negative comments
    negative_comments = test.loc[test['predicted_label']
                                 == 'negative', 'text'].head(5).tolist()

    response = {
        'positive_count': int(positive_count),
        'negative_count': int(negative_count),
        'positive_comments': positive_comments,
        'negative_comments': negative_comments,
        'positive_percentage': int(positive_percentage),
        'negative_percentage': int(negative_percentage)
    }

    return jsonify(response)


@app.route('/predict_single', methods=['GET'])
def predict_for_one():
    # Get the text to predict from the request
    text = request.args.get('text', '')

    # Preprocess the text
    cleaned_text = preprocess_text(text)

    # Vectorize the text
    text_vector = vectorizer.transform([cleaned_text])

    # Predict the sentiment
    prediction = classifier.predict(text_vector)[0]

    # Return the prediction
    return jsonify({'sentiment': prediction})


def preprocess_text(text):
    # Your preprocessing steps here
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join([w for w in text.split() if len(w) > 1])
    tokenized_text = nltk.word_tokenize(text)
    tokenized_text = nltk.pos_tag(tokenized_text)
    tokenized_text = [(word, tag) for (word, tag) in tokenized_text if tag not in (
        'IN', 'NNP', 'NNPS', 'TO', 'UH', 'CC', 'MD', 'PRP', 'PRP$', 'DT', 'WDT', 'WP', 'WP$', 'WRB')]
    tokenized_text = [ss.stem(word) for word, _ in tokenized_text]
    return ' '.join(tokenized_text)


model_path = 'working/best_model_30k.h5'
model = load_model(model_path)

# Load VGG16 model for feature extraction
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Load tokenizer
with open('working/tokenizer_30k.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load maximum caption length
max_length = 74  # Update this with the actual maximum caption length
# max_length = 35

# Load feature mapping
with open('working/features_30k.pkl', 'rb') as f:
    features = pickle.load(f)

# Define function to generate captions for new images


def generate_caption_image(image_path):
    # Load the image
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    # Extract features
    feature = vgg_model.predict(image, verbose=0)

    # Predict caption
    caption = predict_caption_image(model, feature, tokenizer, max_length)

    return caption

# Define function to predict caption


def predict_caption_image(model, image_feature, tokenizer, max_length):
    # Add start tag for generation process
    in_text = 'startseq'

    # Define function to convert index to word
    def idx_to_word(integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    # Iterate over the max length of sequence
    for i in range(max_length):
        # Encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad the sequence
        sequence = pad_sequences([sequence], maxlen=max_length)
        # Predict next word
        yhat = model.predict([image_feature, sequence], verbose=0)
        # Get index with high probability
        yhat = np.argmax(yhat)
        # Convert index to word
        word = idx_to_word(yhat, tokenizer)
        # Stop if word not found
        if word is None:
            break
        # Append word as input for generating next word
        in_text += " " + word
        # Stop if we reach end tag
        if word == 'endseq':
            break

    return in_text


@app.route('/predict_image', methods=['POST'])
def predict_for_an_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Load the uploaded image
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    # Save the image to a temporary file
    temp_image_path = 'temp.jpg'
    image_file.save(temp_image_path)

    image_path = 'boys.jpeg'
    caption = generate_caption_image(temp_image_path)
    caption = caption.replace('startseq', '').replace('endseq', '').strip()

    blob = TextBlob(caption)

    # Analyze sentiment
    sentiment_score = blob.sentiment.polarity

    # Print sentiment label
    if sentiment_score > 0:
        sentiment_img = "positive"
    elif sentiment_score < 0:
        sentiment_img = "negative"
    else:
        sentiment_img = "neutral"

    # Predict the sentiment

    response = {
        'caption': caption,
        'sentiment': sentiment_img
    }
    os.remove(temp_image_path)
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)


# classifiers = {
#     'Decision Tree': DecisionTreeClassifier(),
#     'Random Forest': RandomForestClassifier(),
#     'Support Vector Machine': SVC(),
#     'Logistic Regression': LogisticRegression(),
#     'Multinomial Naive Bayes': MultinomialNB(),
#     'Gradient Boosting': GradientBoostingClassifier()
# }


# def get_sentiment_label(score):
#     if score > 0:
#         return 'positive'
#     elif score == 0:
#         return 'neutral'
#     else:
#         return 'negative'


# def calculate_accuracy(predicted_labels, sentiment_labels):
#     correct = sum(predicted == sentiment for predicted,
#                   sentiment in zip(predicted_labels, sentiment_labels))
#     total = len(sentiment_labels)
#     return correct / total

# combine['Sentiment_Label'] = combine['Tidy_Tweets'].apply(
#     lambda x: get_sentiment_label(TextBlob(x).sentiment.polarity))

# accuracy_results = {}
# for clf_name, clf in classifiers.items():
#     # Train classifier
#     clf.fit(X_train, y_train)
#     # Predict labels
#     predicted_labels = clf.predict(X_test)
#     # Calculate accuracy
#     accuracy = calculate_accuracy(
#         predicted_labels, combine.loc[train.shape[0]:, 'Sentiment_Label'])
#     accuracy_results[clf_name] = accuracy

# print('\nAccuracy Results:')
# for clf_name, accuracy in accuracy_results.items():
#     print(f'{clf_name} Accuracy:', accuracy)
