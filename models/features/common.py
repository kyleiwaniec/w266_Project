import pickle

import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
import scipy.sparse

datasets = {}

def load_dataset(dataset_name):
    global datasets
    if dataset_name not in datasets:
        print "Loading %s dataset." % dataset_name
        with open("/usr/src/app/data/%s.pickle" % dataset_name, "rb") as f:
            datasets[dataset_name] = pickle.load(f)
    return datasets[dataset_name]

def extract_features(dataset_name, features_def, features_name):
    dataset = load_dataset(dataset_name)
    
    print "Training feature extractor %s." % features_name
    
    features = {}
    features["train_X"] = features_def.train(dataset["train_data"])
    features["train_Y"] = dataset["train_data"]["hasVotes"]
    
    print "Generating validation set..."
    features["validate_X"] = features_def.transform(dataset["validate_data"])
    features["validate_Y"] = dataset["validate_data"]["hasVotes"]
    
    print "Generating test set..."
    features["test_X"] = features_def.transform(dataset["test_data"])
    features["test_Y"] = dataset["test_data"]["hasVotes"]
    
    print "Writing to disk..."
    with open("/usr/src/app/model-data/%s.pickle" % features_name, "wb") as f:
        pickle.dump(features, f)
        
    print "Done."
    return features
    
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
    predictions = {}    
    print "Training models."
    for model_name, model in model_types():
        model.fit(features['train_X'], features['train_Y'])
        predictions[model_name] = model.predict(features['validate_X'])
        score = model.score(features['validate_X'], features['validate_Y'])
        print "## %20s %15s accuracy: %0.1f %%" % (model_name, features_name, score * 100)
    return (predictions, features)   
        
def test_combined_features(feature_names):
    in_features = []
    for features_name in feature_names:
        print "Loading features %s." % features_name
        with open("/usr/src/app/model-data/%s.pickle" % features_name, "rb") as f:
            in_features.append(pickle.load(f))
            
    print "Combining features."

    features = {
        "train_X": scipy.sparse.hstack([f["train_X"] for f in in_features]),
        "train_Y": in_features[0]["train_Y"],
        "validate_X": scipy.sparse.hstack([f["validate_X"] for f in in_features]),
        "validate_Y": in_features[0]["validate_Y"],
    }
    
    combined_name = "_".join(feature_names)
        
    print "Training models."
    for model_name, model in model_types():
        model.fit(features['train_X'], features['train_Y'])
        score = model.score(features['validate_X'], features['validate_Y'])
        print "## %20s %30s accuracy: %0.1f %%" % (model_name, combined_name, score * 100)