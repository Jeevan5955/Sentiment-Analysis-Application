from django.shortcuts import render, redirect
import pickle
from django.contrib import messages
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV
# Create your views here.


file_name = './data/sentiment/books_small_10000.json'


def index(request):

    if request.method == 'POST' and 'analyze' in request.POST:

        test_1 = []
        test_1.append((request.POST.get('message')))

        class Sentiment:
            NEGATIVE = "NEGATIVE"
            NEUTRAL = "NEUTRAL"
            POSITIVE = "POSITIVE"

        class Review:
            def __init__(self, text, score):
                self.text = text
                self.score = score
                self.sentiment = self.get_sentiment()

            def get_sentiment(self):
                if self.score <= 2:
                    return Sentiment.NEGATIVE
                elif self.score == 3:
                    return Sentiment.NEUTRAL
                else:  # Score of 4 or 5
                    return Sentiment.POSITIVE

        class ReviewContainer:
            def __init__(self, reviews):
                self.reviews = reviews

            def get_text(self):
                return [x.text for x in self.reviews]

            def get_sentiment(self):
                return [x.sentiment for x in self.reviews]

            def evenly_distribute(self):
                negative = list(filter(lambda x: x.sentiment ==
                                       Sentiment.NEGATIVE, self.reviews))
                positive = list(filter(lambda x: x.sentiment ==
                                       Sentiment.POSITIVE, self.reviews))
                positive_shrunk = positive[:len(negative)]
                self.reviews = negative + positive_shrunk
                random.shuffle(self.reviews)

        reviews = []
        with open(file_name) as f:
            for line in f:
                review = json.loads(line)
                reviews.append(Review(review['reviewText'], review['overall']))

        training, test = train_test_split(
            reviews, test_size=0.33, random_state=42)

        train_container = ReviewContainer(training)

        test_container = ReviewContainer(test)

        train_container.evenly_distribute()
        train_x = train_container.get_text()
        train_y = train_container.get_sentiment()

        test_container.evenly_distribute()
        test_x = test_container.get_text()
        test_y = test_container.get_sentiment()

        vectorizer = TfidfVectorizer()
        train_x_vectors = vectorizer.fit_transform(train_x)

        test_x_vectors = vectorizer.transform(test_x)

        clf_svm = svm.SVC(kernel='linear')

        clf_svm.fit(train_x_vectors, train_y)

        parameters = {'kernel': ('linear', 'rbf'), 'C': (1, 4, 8, 16, 32)}

        svc = svm.SVC()
        clf = GridSearchCV(svc, parameters, cv=5)
        clf.fit(train_x_vectors, train_y)

        with open('./models/sentiment_classifier.pkl', 'wb') as f:
            pickle.dump(clf, f)

        with open('./models/sentiment_classifier.pkl', 'rb') as f:
            loaded_clf = pickle.load(f)

        new_test = vectorizer.transform(test_1)

        messages.info(request, loaded_clf.predict(new_test)[0])

    elif request.method == 'POST' and 'analyze' in request.POST:

        pass

    return render(request, 'index.html')
