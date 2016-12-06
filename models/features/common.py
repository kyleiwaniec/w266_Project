import datetime
import json
import pickle
import random

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

def extract_features(dataset_name, features_def, features_name, sampling=1.0):
    dataset = load_dataset(dataset_name)

    print "Training feature extractor %s." % features_name

    features = {}
    train_data_size = np.shape(dataset["train_data"])[0]
    train_indices = random.sample(
        range(train_data_size),
        int(train_data_size*sampling))
    features["train_X"] = features_def.train(dataset["train_data"].iloc[train_indices])
    features["train_Y"] = dataset["train_data"].iloc[train_indices]["label"]

    print "Generating validation set..."
    features["validate_X"] = features_def.transform(dataset["validate_data"])
    features["validate_Y"] = dataset["validate_data"]["label"]

    print "Generating test set..."
    features["test_X"] = features_def.transform(dataset["test_data"])
    features["test_Y"] = dataset["test_data"]["label"]

    print "Writing to disk..."
    with open("/usr/src/app/model-data/%s.pickle" % features_name, "wb") as f:
        pickle.dump(features, f)

    print "Done."

def model_types():
    return [
        ["MultinomialNB", MultinomialNB()],
        ["LinearSVC", LinearSVC()],
        ["MLP", MLPClassifier(hidden_layer_sizes=(20,20), early_stopping=True)],
        ["MLP2", MLPClassifier(hidden_layer_sizes=(40,40), early_stopping=True)],
    ]

def load_features(features_name):
    print "Loading features %s." % features_name
    with open("/usr/src/app/model-data/%s.pickle" % features_name, "rb") as f:
        features = pickle.load(f)
    return features

def update_table(model_name, features_name, precision, recall):
    try:
        with open("/usr/src/app/model-data/results.json", "r") as f:
            table = json.load(f)
    except:
        table = {}
    
    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0
    table["%s:%s" % (model_name, features_name)] = [
        model_name, features_name, precision, recall, f1, datetime.datetime.now().isoformat()]

    with open("/usr/src/app/model-data/results.json", "w") as f:
        json.dump(table, f)
    
def test_features(features_name):
    features = load_features(features_name)

    print "Training models."
    for model_name, model in model_types():
        model.fit(features['train_X'], features['train_Y'])
        preds = model.predict(features['validate_X'])
        
        true_positives = float(np.sum((features['validate_Y'] == True) & (preds == True)))
        true_negatives = float(np.sum((features['validate_Y'] == False) & (preds == False)))
        false_positives = float(np.sum((features['validate_Y'] == False) & (preds == True)))
        false_negatives = float(np.sum((features['validate_Y'] == True) & (preds == False)))
        
        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, true_positives + false_negatives)
        print "## %20s %15s precision: %0.1f%% recall: %0.1f%%" % (
            model_name, features_name, precision * 100, recall * 100)
        
        update_table(model_name, features_name, precision, recall)
        
def show_errors(dataset_name, features_name, model_name):
    dataset = load_dataset(dataset_name)
    features = load_features(features_name)

    _, model = next(tup for tup in model_types() if tup[0] == model_name)
    print "Training model %s." % model_name
    model.fit(features['train_X'], features['train_Y'])
    
    preds = model.predict(features['validate_X'])
    
    num_examples = np.shape(preds)[0]
    true_positives = (features['validate_Y'] == True) & (preds == True)
    true_negatives = (features['validate_Y'] == False) & (preds == False)
    false_positives = (features['validate_Y'] == False) & (preds == True)
    false_negatives = (features['validate_Y'] == True) & (preds == False)
    
    print "## True positives: %f" % (float(np.sum(true_positives)) / num_examples)
    print "## True negatives: %f" % (float(np.sum(true_negatives)) / num_examples)
    print "## False positives: %f" % (float(np.sum(false_positives)) / num_examples)
    print "## False negatives: %f" % (float(np.sum(false_negatives)) / num_examples)
    
    print "True positives:"
    print dataset['validate_data']['content'].iloc[np.where(true_positives)][:25]
    print "True negatives:"
    print dataset['validate_data']['content'].iloc[np.where(true_negatives)][:25]
    print "False positives:"
    print dataset['validate_data']['content'].iloc[np.where(false_positives)][:25]
    print "False negatives:"
    print dataset['validate_data']['content'].iloc[np.where(false_negatives)][:25]
  
 

def load_combined_features(feature_names):
    in_features = []
    for features_name in feature_names:
        print "Loading features %s." % features_name
        with open("/usr/src/app/model-data/%s.pickle" % features_name, "rb") as f:
            in_features.append(pickle.load(f))

    print "Combining features: %s" % " + ".join([str(np.shape(f["train_X"])) for f in in_features])

    features = {
        "train_X": scipy.sparse.hstack([f["train_X"] for f in in_features]),
        "train_Y": in_features[0]["train_Y"],
        "validate_X": scipy.sparse.hstack([f["validate_X"] for f in in_features]),
        "validate_Y": in_features[0]["validate_Y"],
    }
    return features

def test_combined_features(feature_names):
    features = load_combined_features(feature_names)
    combined_name = "_".join(feature_names)

    print "Training models."
    for model_name, model in model_types():
        model.fit(features['train_X'], features['train_Y'])
        preds = model.predict(features['validate_X'])
        
        true_positives = float(np.sum((features['validate_Y'] == True) & (preds == True)))
        true_negatives = float(np.sum((features['validate_Y'] == False) & (preds == False)))
        false_positives = float(np.sum((features['validate_Y'] == False) & (preds == True)))
        false_negatives = float(np.sum((features['validate_Y'] == True) & (preds == False)))
        
        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, true_positives + false_negatives)
        print "## %20s %15s precision: %0.1f%% recall: %0.1f%%" % (
            model_name, combined_name, precision * 100, recall * 100)
        
        update_table(model_name, combined_name, precision, recall)
