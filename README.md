The steps to perform hyperparameter tuning tasks are underlined as below:

GOAL: The evaluation metrics of the cWGAN-GP model is the chi-squared/degrees of freedom, the p-value of the hypothesis that Geant4 and BoloGAN are the same. 

Workflow:

1. Cross Validation: Divide the data into folds.
2. Train and validate using chi sqaured/ndf and p-value on each fold. Aggregate metrics across folds.
3. Tune regularization strengths during cross-validation. In our case, it is gradient penalty.
4. Iterate over hyperparameters. Optimise based on cross-validation results for chi-squared/ndf p-value.




