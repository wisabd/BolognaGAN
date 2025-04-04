# Problem Statement
Test the following hypothesis:
"The energy distribution of the single particle showers produced by CWGAN-GP is statistically indistinguishable from that of Geant4, meaning the generated data accurately represents Geant4 showers"

# Why is it important?
Simulation is essential requirement for comparing with actual data from CERN for analysis of high energy experiments. Traditionally simulation is done by Geant4, a resource intensive tool.
This model conditional WGAN-gradient penalty created at University of Bologna is a fast, lightweight alternate to traditional resource intensive simulation tool (Geant4) for the particle physics ATLAS experiment at CERN in Geneva, Switzerland. 

# Goal
The project compares two distributions and evaluates the goodness of fit with reduced chi squares χ²/ndf as the metric. 
One distribution is energy distribution Geant4 from dataset  (csvFiles, rootFiles, binning.xml), the other is distribution is generated by the conditional WGAN-GP model.

## Dataset

A binning file containing radial and angular binning and the hyperparameters. Voxels defined and maximized per layer and eta η
<img width="364" alt="image" src="https://github.com/user-attachments/assets/90a8aeac-3aa4-42f5-8d6c-ec81733d9eb5" />

CSV files containing single particle energy depositions flattened voxels

ROOT files containing original histogram containing Geant4 full normalized energy distribution “h_vox”, the statistics of events in ROOT files for every energy level given:
<img width="863" alt="image" src="https://github.com/user-attachments/assets/2031dc98-5d7d-4fb1-a4a3-3af9d24aaf41" />

ROOT file structure
<img width="252" alt="Screenshot 2025-02-24 123818" src="https://github.com/user-attachments/assets/0974bc35-bf02-4046-8034-297bf58a5dfe" />


## A brief outline of BoloGAN
+ Model: conditional Wasserstein GAN with Gradient Penalty
+ cWGAN-GP model
+ <img width="522" alt="Screenshot 2025-02-18 060956" src="https://github.com/user-attachments/assets/7ad32fbe-d213-438d-9925-ff8e1b1205dc" />
+ Loss function: <img width="135" alt="image" src="https://github.com/user-attachments/assets/bd062795-3394-4f93-8608-7403ed26755e" />
+ Final objective function after gradient penalty <img width="328" alt="image" src="https://github.com/user-attachments/assets/66ec8062-4de5-44df-8ff4-b67850604502" />

## Methodology

+ The goal is to implement hypothesis testing (chi-squared test) between two distributions with the following null hypothesis: Null Hypothesis (H0): “Energy distribution of single-particle pion shower generated by BoloGAN is statistically indistinguishable from that produced by Geant4 Full Simulation.”

+ Generate energy distribution in voxels from the generator and normalize the energy.

+ Plot the normalized energy distribution in GeV (with hits “entries” on the y-axis) as histogram “h_gan” via ROOT.TH1F.

+ Retrieve the original histogram containing Geant4 full normalized energy distribution “h_vox” from the ROOT files for comparison.

+ Evaluate the chi-squared test using chi2test(h_vox, h_gan) to obtain chi² and ndf (number of degrees of freedom).

+ Visualize the results via TensorBoard.
  
+ Pass hyperparameters through binning.xml file to the BoloGAN model

+ Execute Bash scripts to run train.py to define data, user and home directories along with particle id, η range, epoch numbers and the binning.xml file

+ Utilize LXPLUS batch at CERN and conduct runs by submitting jobs to HTCondor pool, using NVIDIA H100/A100/V100 GPUs

+ Conducted preliminary runs for selecting best performing models with best generator architecture

+ Performed systematic random search for optimizing hyperparameters of batch size, learning rate and Discriminator/Generator ratio

+ Trained each model for multiple (minimum 3) runs

+ Logged the results and reported the Tensorboard visualizations of the  models with lowest chi squared/ndf 

## My task
I optimized the following hyperparameters:
- Generator neural network nodes (width and depth)
- Batch size
- Discriminator/Generator ratio
- Learning rate
- Gradient Penalty (to mitigate overfitting)

## Screenshots of the project
Result of the model at the start of the oroject
![Best-reducedchi2-Pions](https://github.com/user-attachments/assets/cc81a3ae-6c27-48f7-b9f9-a07359e1f4dd)

Final result of the model after the project (Improvement: 28%)
<img width="448" alt="Best-BigModel" src="https://github.com/user-attachments/assets/6ca5e76f-3342-41d1-aec4-ff2f8fd65742" />










 






