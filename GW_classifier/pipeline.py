from argparse import ArgumentParser
from GW_classifier.dataprocessing import load_and_preprocess
from GW_classifier.training import train_model 
from GW_classifier.testing import test_model 
from GW_classifier.utils import load_module, create_directory 
from GW_classifier.logger import get_logger
from sklearn.model_selection import train_test_split 
import numpy as np 
import json 


logger = get_logger(__name__)

def main(config): 
    #Start logger and pipeline 
    seed = config.get('seed', None)
    np.random.seed(seed) 
    logger.info(f"Starting the training pipeline with seed: {seed}")

    #Load and scale the data 
    X, y = load_and_preprocess(config['datasource'])
    scaler = load_module("sklearn.preprocessing", config['scaler'])
    X = scaler.fit_transform(X) 

    #Split data and train the model 
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.05, random_state=config['seed'])
    model, results = train_model(config, X_train, y_train) 
    logger.info("Model training completed, starting testing phase")

    #Test model and save the fitted results 
    test_model(config, model, X_test, y_test)

    return 0 


if __name__ == '__main__': 
    parser = ArgumentParser(description="KNN Classifier Configuration")
    parser.add_argument('Config', type=str, help='Path to the configuration file') 
    arg = parser.parse_args()
    with open(arg.Config, 'r') as file:
        config = json.load(file)
    main(config)