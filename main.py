import sys 
from argparse import ArgumentParser
sys.path.append("..")  # Adjust the path to import from the parent directory
from GW_classifier.neuralnetwork import FFNetwork
from GWclassifier.training import train_network
from GW_classifier.dataprocessing import preprocessData
from GW_classifier.metrics import auc_metric
from GW_classifier.training import train_network    
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metric import roc_auc_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np  
from seaborn import heatmap 
from sklean.neighbors import KNeighborsClassifier 
import json 




    


def main():    # Read the data
    df = readfile("data/merged_data.root")
    
    # Preprocess the data
    df = preprocessData(df)
    
    # Split the data into features and target
    X = df.drop(columns=['class'])
    y = df['class']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Cross validation for KNN 
    KNN = KNeighborsClassifier() 
    knn_params = {'n_neighbors': np.arange(1, 201, 50).astype(int),}
    GridSearchCV_knn = GridSearchCV(KNN, knn_params, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1)
    GridSearchCV_knn.fit(X_train_scaled, y_train) 
    print(f"Best KNN parameters: {GridSearchCV_knn.best_params_}")
    print(f"Best KNN score: {GridSearchCV_knn.best_score_}")
    # Train the KNN model  
    knn_predictions = GridSearchCV_knn.predict(X_test_scaled) 
    

    # Train the neural network
    hidden_layers = []
    model = FFNetwork(input_dim=X_train_scaled.shape[1], hidden_layers=[64, 32], output_dim=2, activation='relu', dropout=0.5)
    results = train_network(model, X_train_scaled, y_train, epochs=50, batch_size=32)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    
    # Compute AUC
    auc_value = auc_metric(y_test, y_pred)
    
    print(f"AUC: {auc_value}")


if __name__ == "__main__": 
    parser = argparse.ArgumentParse()
    parser.add_argument("--config", default = "config.yaml", help = "path to config file for the analysis")
    parser.add_argument("--mode", default = "train", help = "Mode in which running the code.  Available options are train or test. Default is train") 
    args = parser.parse_args()  
    with open(config, 'r') as f: 
        config=  json.load(f)

    main(args.config, args.mode) 