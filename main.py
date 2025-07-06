from argparse import ArgumentParser
from GW_classifiers.training import train_model 
from GW_classifiers.dataprocessing import loadAndPreprocess
from GW_classifiers.metrics import auc_metric
from GW_classifiers.training import train_model     
from sklearn.model_selection import train_test_split
import numpy as np  
import json 


    
def main(config, mode = "Train"): 
    X, y = loadAndPreprocess(config['datasource'])
    print(X.shape)
    import sys
    sys.exit() 
    scaler = load_module('sklearn.preprocessing', config['scaler'])
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = config["train_size"])
    
    for model_name in config["models_to_run"]: 
        model_config = config["models"][model_name]
        #model = train_model(model_name, X_train, y_train, config[model_name])




if __name__ == "__main__": 
    parser = ArgumentParser()
    parser.add_argument("--config", default = "config.json", help = "path to config file for the analysis")
    parser.add_argument("--mode", default = "train", help = "Mode in which running the code.  Available options are train or test. Default is train") 
    args = parser.parse_args()  
    with open(args.config, 'r') as f: 
        config=  json.load(f)

    main(config, args.mode) 