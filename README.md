**Hyperparameter Optimization**

GOAL: The evaluation metric of the hyperparameter tuning of the cWGAN-GP model is the total χ² between Geant4 generated data and that from BoloGAN. 

# Workflow:

1. Train/test split the data.  
2. Iterate over hyperparameters. 
3. Evaluate and log the models' performances according to total χ² metric.
4. The model with lowest χ² is the optimal configuration of hyperparameters.

# Running Order

**Brief Overview of the Project**

On This Repo BoloGAN: contains the GAN training code; BoloGANtainer.def: the container recipe file.

Not on This Repo: BoloGANtainer.sif: the container itself; data: the folder with the training data.

Data
The data is provided by the ATLAS collaboration and is not available on this repository. It has a structure like the following:

data
├── csvFiles
├── rootFiles
└── binning.xml
The csvFiles directory contains the CSV files with the training data.
The rootFiles directory contains the ROOT files with the training data.
The binning.xml file contains the binning information.




