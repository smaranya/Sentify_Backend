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
warnings.filterwarnings("ignore", category=DeprecationWarning)


train = pd.read_csv('train.csv', dtype=str)
# train.columns = ['label', 'text']
train_original = train.copy()

test = pd.read_csv('test.csv', dtype=str)
# test.columns = ['text']
test_original = test.copy()

# combine = train.append(test, ignore_index=True, sort=True)
combine = pd.concat([train, test], ignore_index=True, sort=True)
combine['text'] = combine['text'].astype(str)


def remove_pattern(text, pattern):
    r = re.findall(pattern, text)
    for i in r:
        text = re.sub(i, "", text)
    return text


# remove user mentions
combine['Tidy_Tweets'] = np.vectorize(
    remove_pattern)(combine['text'], "@[\w]*")

# remove links
combine['Tidy_Tweets'] = combine['Tidy_Tweets'].apply(
    lambda x: re.sub(r'https?://\S+', ' ', x))

# remove punctuations
# combine['Tidy_Tweets'] = combine['Tidy_Tweets'].str.replace("[^a-zA-Z#]", "")
combine['Tidy_Tweets'] = combine['Tidy_Tweets'].apply(
    lambda x: re.sub(r'[^\w\s]', ' ', x))

# remove short words
combine['Tidy_Tweets'] = combine['Tidy_Tweets'].apply(
    lambda x: ' '.join([w for w in x.split() if len(w) > 1]))

# tokenize
# tokenized_text = combine['Tidy_Tweets'].apply(lambda x: x.split())
tokenized_text = combine['Tidy_Tweets'].apply(lambda x: nltk.word_tokenize(x))

tokenized_text = tokenized_text.apply(lambda x: nltk.pos_tag(x))

tokenized_text = tokenized_text.apply(lambda line: [(word, tag) for (word, tag) in line if tag not in (
    'IN', 'NNP', 'NNPS', 'TO', 'UH', 'CC', 'MD', 'PRP', 'PRP$', 'DT', 'WDT', 'WP', 'WP$', 'WRB')])

tokenized_text = tokenized_text.apply(
    lambda line: [word for (word, _) in line])

# stemming
ss = SnowballStemmer("english")

tokenized_text = tokenized_text.apply(lambda x: [ss.stem(i) for i in x])

for i in range(len(tokenized_text)):
    tokenized_text[i] = ' '.join(tokenized_text[i])

combine['Tidy_Tweets'] = tokenized_text

# Bag-of-Words
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(combine['Tidy_Tweets'])

# Split the combined dataset into training and test sets
X_train = X[:train.shape[0]]
y_train = train['sentiment']
X_test = X[train.shape[0]:]

# Train a Decision Tree classifier
classifier = GradientBoostingClassifier()
classifier.fit(X_train, y_train)

# Predict labels for the test set
predicted_labels = classifier.predict(X_test)

test_reset_index = test.reset_index(drop=True)

# # Add predicted labels to the combined dataset
combine.loc[train.shape[0]:, 'predicted_label'] = predicted_labels

# print(combine[['Tidy_Tweets', 'predicted_label']].tail())
# Assuming combine dataframe contains the predicted labels in 'predicted_label' column for the test set
# Count the number of positive and negative comments
positive_count = combine.loc[train.shape[0]:,
                             'predicted_label'].value_counts().get('positive', 0)
negative_count = combine.loc[train.shape[0]:,
                             'predicted_label'].value_counts().get('negative', 0)

# Calculate the total number of comments
total_comments = positive_count + negative_count

# Calculate the percentage of positive and negative comments
positive_percentage = round((positive_count / total_comments) * 100)
negative_percentage = round((negative_count / total_comments) * 100)

print("Percentage of Positive Comments:", positive_percentage)
print("Percentage of Negative Comments:", negative_percentage)

# Filter out the first five positive comments
positive_df = combine.loc[combine['predicted_label']
                          == 'positive', ['text']].head(5)

# Filter out the first five negative comments
negative_df = combine.loc[combine['predicted_label']
                          == 'negative', ['text']].head(5)

# Rename the columns for better understanding
positive_df.columns = ['Positive Comments']
negative_df.columns = ['Negative Comments']

# Reset index for better visualization
positive_df.reset_index(drop=True, inplace=True)
negative_df.reset_index(drop=True, inplace=True)

# Display the dataframes
print("First five positive comments:")
print(positive_df)
print("\nFirst five negative comments:")
print(negative_df)


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
