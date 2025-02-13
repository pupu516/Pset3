import os
import numpy as np
import matplotlib.pyplot as plt

# Input directory
input_dir = '/root/Desktop/yambo/Pset3/Archive_P2/Local_density_of_states_near_band_edge'

# List all relevant text files
text_files = sorted([f for f in os.listdir(input_dir)
                     if f.startswith("local_density_of_states_for_level_") and f.endswith(".txt")])

# Parameters for the sub-region (centered, 5x5 window)
sub_size = 5  # Define sub-region size
avg_ldos = []  # Store average LDOS for each level
indices = []  # Store corresponding level indices

# Iterate through each file
for file_name in text_files:
    file_path = os.path.join(input_dir, file_name)
    
    # Load the data from the text file
    try:
        data = np.loadtxt(file_path, delimiter=",")
    except ValueError:
        data = np.loadtxt(file_path)

    # Get matrix dimensions
    rows, cols = data.shape
    center_x, center_y = rows // 2, cols // 2  # Determine center coordinates

    # Ensure the sub-region doesn't go out of bounds
    x_start = max(0, center_x - sub_size // 2)
    x_end = min(rows, center_x + sub_size // 2)
    y_start = max(0, center_y - sub_size // 2)
    y_end = min(cols, center_y + sub_size // 2)

    # Extract the sub-region
    sub_region = data[x_start:x_end, y_start:y_end]

    # Compute the average LDOS in the sub-region
    avg_value = np.mean(sub_region)
    avg_ldos.append(avg_value)
    
    # Extract the file index for plotting
    file_index = int(file_name.split("_")[-1].split(".")[0])
    indices.append(file_index)

# Ensure data is sorted correctly
indices, avg_ldos = zip(*sorted(zip(indices, avg_ldos)))

# Plot the results with improved settings
plt.figure(figsize=(8, 5))
plt.plot(indices, avg_ldos, marker='o', linestyle='-', color='b', linewidth=2, markersize=8, label='Average LDOS')
plt.xlabel("Level Index", fontsize=12)
plt.ylabel("Average LDOS in Sub-Region", fontsize=12)
plt.title("Evolution of Local Density of States in Selected Sub-Region", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Save the figure
output_plot = os.path.join(input_dir, "average_ldos_evolution.png")
plt.savefig(output_plot, dpi=300, bbox_inches="tight")
plt.show()

print("Analysis complete. Average LDOS evolution plot saved.")

