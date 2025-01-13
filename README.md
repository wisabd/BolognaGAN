**Hyperparameter Optimization**

GOAL: The evaluation metrics of the cWGAN-GP model is the chi-squared/degrees of freedom, the p-value of the hypothesis that Geant4 and BoloGAN are the same. 

# Workflow:

2. Train and validate using chi sqaured/ndf and p-value on each fold. Aggregate metrics.
3. Tune regularization strengths during cross-validation. In our case, it is gradient penalty.
4. Iterate over hyperparameters. 

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




