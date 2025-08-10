# GW_classifiers

## The Author
Alessandro Martini, PhD student in physics, martini.alessandr@gmail.com 

## The Problem  

Gravitational-wave detectors are extremely sensitive instruments, capable of capturing the faint ripples in spacetime caused by distant astrophysical events.  

A wide range of data analysis pipelines scan the detector output for signals — including those that are not yet observed and may not match predictions from general relativity. These searches are known as **"burst" searches**.  

In burst searches, one of the key challenges is **distinguishing genuine astrophysical signals from transient instrumental noise**. Complex frameworks such as *Coherent WaveBurst (cWB)* [1] analyze the data and generate triggers for *candidate* gravitational-wave events. However, even after this step, reliably separating real signals from noise remains a difficult task.  

This is where **machine learning** can help. By using features generated from simulated events, a classifier can be trained to tell apart true gravitational-wave signals from noise artifacts. This pipeline aims to take a step forward in that direction — providing a reproducible, flexible framework for testing and applying machine learning methods to this problem.  

## The Data  

The data used in this project are produced by *Coherent WaveBurst* (cWB) as **triggers** from its detection pipeline. These triggers require further study to be classified into one of two classes:  
- **[0]**: Noise transients  
- **[1]**: Genuine gravitational-wave signals  

For each candidate event, cWB outputs a set of **features** describing the event’s properties. This naturally lends itself to a **supervised learning** approach:  
- Instances from the **noise** class (`0`) are generated using the *time-slides method* [2]  
- Instances from the **signal** class (`1`) are generated using simulations  

All datasets are obtained by running cWB on public gravitational-wave data. They are stored and linked in the `Data` folder of this repository.  

---

## How to Use It  

The pipeline is designed to be **easy to run** and **highly configurable**.  
Configuration files in the `Config` directory allow you to fully customize the analysis.  

Clone the repository, prepare your configuration file, and run:  

```console
python -m GW_classifier.pipeline Config/<config_name>.json
```

## How to analyse new data with pre-trained models 
---- To be implemented! ----- 

## The Pipeline and Methods — Overview  

Given a tabular dataset with features and labels, this problem can be framed as a **binary classification task**.  
Because the feature space is complex and far from linear, advanced **non-linear classifiers** are required for reliable predictions.  

This pipeline is designed to be **highly configurable**:  
- The `Config` folder allows you to select **any classifier** available in *scikit-learn*  
- A customizable **PyTorch neural network** implementation is included in the `NeuralNetwork` module  

### What You Can Configure  
Using the configuration file, you can:  
- **Choose the ML method** (e.g., Random Forest, KNN, Neural Network)  
- **Set up cross-validation** parameters  
- **Define evaluation metrics**  
- **Save results** to a dedicated directory, enabling the fitted model to be reused for predicting labels on new incoming data  

Several ready-to-use configuration files are already available in the `Config` folder.

---

## Using Pre-Trained Models on New Data  

> **Status**: To be implemented  

This feature will allow the pipeline to load a previously trained model and apply it directly to classify new data, without retraining.


## Repository Structure

For readability and maintainability, the pipeline is divided into different modules, each handling a specific part of the workflow:

GW_classifier/
├── Config/ # Configs to customize analysis
├── Data/ # Data to be analyzed
├── Results/ # Results, plots, pretrained models
├── Notebooks/ # Exploratory or experimental notebooks
└── GW_classifier/
├── init.py
├── logger.py # Logging setup
├── utils.py # Load/save models and helpers
├── dataprocessing.py # Load and process data
├── NeuralNetwork.py # PyTorch feed-forward network
├── plot.py # Plotting results
├── training.py # Training logic + CrossValidation
├── testing.py # Final evaluation stage
└── pipeline.py # Orchestrates the pipeline




# References 
[1] Marco Drago, et Al. **coherent WaveBurst, a pipeline for unmodeled gravitational-wave data analysis**, SoftwareX, Volume 14, 2021,(https://www.sciencedirect.com/science/article/pii/S2352711021000236)
[2] 
[3] 
