### Description
The project compares two distributions and evaluates the goodness of fit with reduced chi squares χ²/ndf as the metric. 
One distribution is energy distribution Geant4 from dataset  (csvFiles, rootFiles, binning.xml), the other is distribution is generated by the conditional WGAN-GP model.

## cWGAN-GP model
<img width="522" alt="Screenshot 2025-02-18 060956" src="https://github.com/user-attachments/assets/7ad32fbe-d213-438d-9925-ff8e1b1205dc" />


## My task
I optimized the following hyperparameters:
- architecture of the generator (width and depth)
- the latent space which is input into the generator
- the learning rate of the generator discriminator
- the gradient penalty of the model

## Technologies used
- Bash scripts for training on high performance computing systems at LXBATCH at CERN
- Python for data analysis of csv, root, binning.xml files: matplotlib, uproot, pandas, numpy, tensorboard
- Singularity containerization
- HTCondor scripting
- LaTex

## Screenshots of the project
![Best-reducedchi2-Pions](https://github.com/user-attachments/assets/cc81a3ae-6c27-48f7-b9f9-a07359e1f4dd)
![imageData-High12](https://github.com/user-attachments/assets/f8bc903f-efcc-470b-9d3d-0f4102934e1a)
![Loqw12F](https://github.com/user-attachments/assets/09fda5cd-c612-4019-8b18-1b243f18967b)








 






