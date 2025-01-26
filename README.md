**Hyperparameter Optimization**

GOAL:  Build models with different hyperparameters and evaluate them using evaluation metric χ²/ndf between Geant4 generated data and that from BoloGAN.

# Workflow:

0. Study the data
  Data provided to the model is as following:
  ├── csvFiles
  ├── rootFiles
  └── binning.xml
The csvFiles directory contains the CSV files with the training data.
The rootFiles directory contains the ROOT files with the training data.
The binning.xml file contains the binning information.

1. Train/validate/test split the data.
   
2. Define Hyperparameter space. Using Grid search algorithm to pass the hyperparameters to the binning.xml file.
   
3. Run the model. Evaluate Tensorboard graphs of χ²/ndf, generator loss, discriminator loss.
   
4. Detect the model performance, check for overfitting, regularize using suitable method if necessary.
   
5. Evaluate and log the models' performances according to total χ²/ndf metric.
   
6. The model with lowest χ²/ndf is the optimal configuration of hyperparameters.

# Background Information on the project
 






