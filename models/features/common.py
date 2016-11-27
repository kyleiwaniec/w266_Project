import pickle

import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

dataset = None

def load_dataset():
    global dataset
    if not dataset:
        print "Loading ka-comments-balanced dataset."
        with open("/usr/src/app/data/ka-comments-balanced.pickle", "rb") as f:
            dataset = pickle.load(f)        
    return dataset

def extract_features(features_def, features_name):
    dataset = load_dataset()
    
    print "Training feature extractor %s." % features_name
    
    features = {}
    features["train_X"] = features_def.train(dataset["train_data"])
    features["train_Y"] = dataset["train_data"]["hasVotes"]
    features["validate_X"] = features_def.transform(dataset["validate_data"])
    features["validate_Y"] = dataset["validate_data"]["hasVotes"]
    features["test_X"] = features_def.transform(dataset["test_data"])
    features["test_Y"] = dataset["test_data"]["hasVotes"]
    
    with open("/usr/src/app/model-data/%s.pickle" % features_name, "wb") as f:
        pickle.dump(features, f)
        
    print "Done."
    
def model_types():
    return [
        ["MultinomialNB", MultinomialNB()],
        ["LinearSVC", LinearSVC()],
        ["MLP", MLPClassifier(hidden_layer_sizes=(20,20), early_stopping=True)],
    ]
    
def test_features(features_name):
    print "Loading features %s." % features_name
    with open("/usr/src/app/model-data/%s.pickle" % features_name, "rb") as f:
        features = pickle.load(f)
        
    print "Training models."
    for model_name, model in model_types():
        model.fit(features['train_X'], features['train_Y'])
        score = model.score(features['validate_X'], features['validate_Y'])
        print "## %20s %15s accuracy: %0.1f %%" % (model_name, features_name, score * 100)
