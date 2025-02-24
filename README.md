### Description
The project compares two distributions and evaluates the goodness of fit with reduced chi squares χ²/ndf as the metric. 
One distribution is energy distribution Geant4 from dataset  (csvFiles, rootFiles, binning.xml), the other is distribution is generated by the conditional WGAN-GP model.

## cWGAN-GP model
[!model](\media\)

## My task
My responsibility is to optimize the following hyperparameters:
- architecture of the generator
- the latent space which is input into the generator
- the learning rate of the generator discriminator
- the gradient penalty of the model

## Technologies used
- Bash scripts for training on high performance computing systems at LXBATCH at CERN
- Python for data analysis of csv, root, binning.xml files: matplotlib, uproot, pandas, numpy
- Singularity containerization
- HTCondor scripting
- LaTex

## Screenshots of the project






 






