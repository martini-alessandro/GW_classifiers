from argparse import ArgumentParser
from GW_classifier.dataprocessing import load_and_preprocess
from GW_classifier.training import train_model 
from GW_classifier.utils import load_module
from sklearn.model_selection import train_test_split 
import numpy as np 
import json 

def main(config): 
    if 'seed' in config:
        np.random.seed(config['seed'])
    X, y = load_and_preprocess(config['datasource'])
    scaler = load_module("sklearn.preprocessing", config['scaler'])
    X = scaler.fit_transform(X) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.05, random_state=config['seed'])
    model = train_model(config, X_train, y_train) 
    if hasattr(model, "predict"): 
        y_hat = model.predict(X_test)
    else:
        y_hat = model(X_test) 

    return 0 


if __name__ == '__main__': 
    parser = ArgumentParser(description="KNN Classifier Configuration")
    parser.add_argument('Config', type=str, help='Path to the configuration file') 
    arg = parser.parse_args()
    with open(arg.Config, 'r') as file:
        config = json.load(file)
    main(config)