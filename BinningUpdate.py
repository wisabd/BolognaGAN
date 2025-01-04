#BinningUpdate

import xml.etree.ElementTree as ET
import os
from itertools import product

# Define the hyperparameter grid
hyperparameter_space = {
    #"learning_rate": [1e-4],
    #"beta1": [0.1, 0.9],
    "beta": [0.95, 0.9999],
    #"num_hidden_layers": [3, 4],
    "generator": [
        (25, 50, 100),
        (100, 200, 400),
        (200, 400, 800),
        (400, 800, 1600),
        (25, 50, 100, 200),
        (100, 200, 400, 800),
        (200, 400, 800, 1600),
        (400, 800, 1600, 3200),
    ],
    #"latent_dim": [(25, 100, 200, 400)],
}

# Function to generate hyperparameter combinations
def generate_combinations(hyperparameter_space):
    keys, values = zip(*hyperparameter_space.items())
    return [dict(zip(keys, combination)) for combination in product(*values)]

# Function to modify the binning.xml file for a specific configuration
def modify_binning_xml(input_file, output_file, hyperparameters):
    tree = ET.parse(input_file)
    root = tree.getroot()

    latent_dim = hyperparameters["generator"][0]
    # Update Pion hyperparameters
    for particle in root.findall("Particle"):
        if particle.get("name") == "pion":
            #particle.set("learningRate", str(hyperparameters["learning_rate"]))
            #particle.set("beta1", str(hyperparameters["beta1"]))
            particle.set("beta", str(hyperparameters["beta"]))
            particle.set("latentDim", str(latent_dim))
            #particle.set("hiddenLayers", str(hyperparameters["num_hidden_layers"]))
            particle.set("generator", ",".join(map(str, hyperparameters["generator"])))

    # Save the modified file
    tree.write(output_file)
    print(f"Modified binning.xml saved as: {output_file}")

# Main function
def main():
    input_file = "C:/Users/Nasreen Akhtar/Downloads/binning (1).xml" 
    output_dir = "C:/Users/Nasreen Akhtar/Downloads/configs"
    os.makedirs(output_dir, exist_ok=True)

    # Generate all hyperparameter combinations
    combinations = generate_combinations(hyperparameter_space)

    # Iterate through combinations
    for i, hyperparameters in enumerate(combinations, start=1):
        print(f"Generating configuration {i}/{len(combinations)}: {hyperparameters}")

        # Modify binning.xml for the current combination
        output_file = os.path.join(output_dir, f"binning_pion_config_{i}.xml")
        modify_binning_xml(input_file, output_file, hyperparameters)

if __name__ == "__main__":
    main()
