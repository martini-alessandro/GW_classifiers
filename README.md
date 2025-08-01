# GW_classifiers

## The Author
Alessandro Martini, PhD student in physics, martini.alessandr@gmail.com 

## The problem 
Gravitational waves detector are very sensitive instruments to detect gravitational waves.\\
The problem with these detectors and analysis pipelines is that a lot of triggers are produced during
an analysis, and distinguishing between genuine signals and noise transients is a problem yet to be 
completely solved. \\ 
Machine Learning Algorithms are advanced tools that allow to perform highly accurate classifications,
the aim of this repo-project is to build different ML classifiers to maximise the capacity of distinguishing
between genuine signals and noise. \\ 
The data used are the triggers produced by the coherent WaveBurst [1] pipeline, together with its features.\\ 
These features are recombined to create predictive features to train different models that are tested on 
a set of noise instances VS signal instances which are buil via proper simulations. \\ 


## The methods 
The repository implements different different classifiers to distinguish between the two classes. Since 
the boundary are probably far from linearity, highly flexible - non linear methods are used: 
- K Nearest Neighbors
- Random Forest
- Feed Forward Network



## Repository structure
GW_Classifiera
|-GW Classifier      #contains core codes 
  |- utils.py        #contains utilities 
|



# References 
[1] Marco Drago, et Al. **coherent WaveBurst, a pipeline for unmodeled gravitational-wave data analysis**, SoftwareX, Volume 14, 2021,(https://www.sciencedirect.com/science/article/pii/S2352711021000236)
[2] 
[3] 
