import numpy as np
import matplotlib.pyplot as plt
import os

# Define the directory path
directory_path = 'local_density_of_states_heatmap'


# List of text files
text_files = [
    'local_density_of_states_for_level_0.txt',
    'local_density_of_states_for_level_1.txt',
    'local_density_of_states_for_level_2.txt',
    'local_density_of_states_for_level_3.txt',
    'local_density_of_states_for_level_4.txt',
    'local_density_of_states_for_level_5.txt',
    'local_density_of_states_for_level_6.txt',
    'local_density_of_states_for_level_7.txt',
    'local_density_of_states_for_level_8.txt',
    'local_density_of_states_for_level_9.txt',
    'local_density_of_states_for_level_10.txt'
]

# Iterate through each file
for i, file_name in enumerate(text_files):
    # Read the data from the text file
    with open(file_name, 'r', encoding='utf-8') as file:
        data = file.read()

    # Replace commas with periods
    data = data.replace(',', '.')

    # Convert the string data to a NumPy array
    data = np.fromstring(data, sep=' ')

    # Reshape the data to a 2D array (assuming square matrices)
    size = int(np.sqrt(len(data)))
    data = data[:size*size].reshape((size, size))

    # Generate the heatmap
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Intensity')

    # Set the title
    plt.title(f'Local Density of States - Level {i}')

    # Save the heatmap as an image
    image_path = os.path.join(directory_path, f'heatmap_level_{i}.png')
    plt.savefig(image_path)
    plt.close()

