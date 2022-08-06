import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers) -> None:
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

documents_f = open("classifiers_pickled/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features_f = open("classifiers_pickled/word_features.pickle", "rb")
word_features:list = pickle.load(word_features_f)
word_features_f.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets_f = open("classifiers_pickled/featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

classifiers_f = open("classifiers_pickled/classifiers.pickle", "rb")
classifiers:list = pickle.load(classifiers_f)
classifiers_f.close()

classifier = classifiers[0]
MNB_classifier = classifiers[1]
BNB_classifier = classifiers[2]
LogisticRegression_classifier = classifiers[3]
SGDClassifier_classifier = classifiers[4]
LinearSVC_classifier = classifiers[5]
NuSVC_classifier = classifiers[6]

voted_classifier = VoteClassifier(classifier, MNB_classifier, LogisticRegression_classifier, SGDClassifier_classifier, NuSVC_classifier, LinearSVC_classifier)
print("voted_classifier accuracy percent: ", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)
