from argparse import ArgumentParser
from GW_classifier.dataprocessing import load_and_preprocess, postprocess_and_plot
from GW_classifier.training import train_model 
from GW_classifier.utils import load_module, create_directory 
from GW_classifier.logger import get_logger
from sklearn.model_selection import train_test_split 
import numpy as np 
import json 


logger = get_logger(__name__)

def main(config): 
    logger.info("Starting the training pipeline")
    if 'seed' in config:
        np.random.seed(config['seed'])
    #Load and scale the data 
    X, y = load_and_preprocess(config['datasource'])
    scaler = load_module("sklearn.preprocessing", config['scaler'])
    X = scaler.fit_transform(X) 

    #Split in train and test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.05, random_state=config['seed'])
    model, results = train_model(config, X_train, y_train) 
    
    #Predict labes for test set 
    y_hat = model.predict(X_test) if hasattr(model, "predict") else model(X_test) 

    postprocess_and_plot(config, model, y_test, y_hat)
    return 0 


if __name__ == '__main__': 
    parser = ArgumentParser(description="KNN Classifier Configuration")
    parser.add_argument('Config', type=str, help='Path to the configuration file') 
    arg = parser.parse_args()
    with open(arg.Config, 'r') as file:
        config = json.load(file)
    main(config)