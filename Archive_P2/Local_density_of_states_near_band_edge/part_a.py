import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Input and output directories
input_dir = '/root/Desktop/yambo/Pset3/Archive_P2/Local_density_of_states_near_band_edge'
output_dir = os.path.join(input_dir, 'local_density_of_states_heatmap')

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get all relevant text files in the input directory
text_files = sorted([f for f in os.listdir(input_dir)
                     if f.startswith("local_density_of_states_for_level_") and f.endswith(".txt")])

# Iterate through each file
for file_name in text_files:
    file_path = os.path.join(input_dir, file_name)
    
    # Load the data from the text file
    try:
        data = np.loadtxt(file_path, delimiter=",")
    except ValueError:
        # If there's a ValueError, try loading without specifying a delimiter
        data = np.loadtxt(file_path)
    
    # Generate the heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(data, cmap="viridis", cbar=True)

    # Set labels and title
    file_index = file_name.split("_")[-1].split(".")[0]  # Extract the level number
    plt.title(f"Local Density of States (Level {file_index})")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # Save the heatmap as an image
    output_path = os.path.join(output_dir, f"heatmap_level_{file_index}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

print("Heatmaps generated and saved successfully.")

