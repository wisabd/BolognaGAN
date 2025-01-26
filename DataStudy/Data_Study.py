import uproot

# Open the ROOT file and tree
file = uproot.open("C:/Users/Wisal Abdullah/Downloads/Data_BoloGAN/data/rootFiles/pions/pid211_E524288_eta_20_25.root")
#tree = file["tree_name"]  # Replace with the actual tree name
print(file.keys()) 


tree = file["rootTree;1"] 
# Print the structure of the tree (branches and leaves)
print(tree.keys())  # Lists all branches in the tree

for branch_name, branch in tree.items():
    print(f"Branch: {branch_name}, Type: {branch.interpretation}")

etot_values = tree["etot"].array()  # Read all 'etot' values as a NumPy array
print(etot_values)
print(len(etot_values))
delta_eta_13 = tree["delta_eta_13"].array()
print(len(delta_eta_13))

# Access branches as arrays
#branch_data = tree["branch_name"].array()  # Replace branch_name with actual branch name
#print(branch_data)
