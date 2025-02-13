import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Input and output directories
input_dir = '/root/Desktop/yambo/Pset3/Archive_P2/Local_density_of_states_near_band_edge'
output_dir = os.path.join(input_dir, 'local density of states height')

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
    
    # Get the dimensions of the data
    rows, cols = data.shape
    
    # Create meshgrid for X and Y axes
    X = np.arange(0, cols)
    Y = np.arange(0, rows)
    X, Y = np.meshgrid(X, Y)
    
    # Create the figure and 3D axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, data, cmap='viridis', edgecolor='none')
    
    # Add a color bar which maps values to colors
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Set labels and title
    file_index = file_name.split("_")[-1].split(".")[0]  # Extract the level number
    ax.set_title(f"Local Density of States (Level {file_index})")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Density")
    
    # Save the figure
    output_path = os.path.join(output_dir, f"height_profile_level_{file_index}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

print("3D surface plots generated and saved successfully.")

