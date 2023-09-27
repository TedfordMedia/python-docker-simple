# advanced_plotting_v2.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def advanced_pixel_plotting_v2(all_pixel_data, output_directory, temperatures_str):
    """
    Plot a variety of visualizations for the given pixel data.

    Args:
    - all_pixel_data (dict): Dictionary containing pixel data for images.
    - output_directory (str): Directory to save the plots.
    - temperatures_str (str): String representation of temperatures for naming.
    """

    # ... [Retaining previous plotting functions here for CDF, Histogram, and Violin plot] ...

    # Heatmap of pixel intensities
    data_matrix = []
    temp_labels = list(all_pixel_data.keys())
    max_pixel_value = max([max(data[0]) for data in all_pixel_data.values()])

    for temperature in temp_labels:
        _, counts, _ = all_pixel_data[temperature]
        full_counts = np.zeros(max_pixel_value + 1)
        full_counts[all_pixel_data[temperature][0]] = counts
        data_matrix.append(full_counts)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data_matrix, cmap='YlGnBu', cbar_kws={'label': 'Frequency'})
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Temperature")
    ax.set_yticklabels(temp_labels)
    ax.set_xticks([0, 255])
    ax.set_xticklabels(['Black', 'White'])
    ax.set_title("Heatmap of Pixel Intensity Frequencies by Temperature")
    plt.tight_layout()
    output_heatmap_file = f"{output_directory}/pixel_intensity_heatmap_{temperatures_str}.png"
    plt.savefig(output_heatmap_file, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {output_heatmap_file}")

    # Boxplot
    all_data = []
    all_temps = []
    for temperature, data in all_pixel_data.items():
        unique_pixels, counts, _ = data
        expanded_data = np.repeat(unique_pixels, counts)
        all_data.extend(expanded_data)
        all_temps.extend([temperature] * len(expanded_data))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=all_temps, y=all_data)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Pixel Intensity")
    ax.set_title("Boxplot of Pixel Intensities by Temperature")
    plt.tight_layout()
    output_boxplot_file = f"{output_directory}/pixel_intensity_boxplot_{temperatures_str}.png"
    plt.savefig(output_boxplot_file, bbox_inches='tight')
    plt.close()
    print(f"Boxplot saved to {output_boxplot_file}")

    # Density Plot (KDE)
    fig, ax = plt.subplots(figsize=(10, 6))
    for temperature, data in all_pixel_data.items():
        unique_pixels, counts, _ = data
        expanded_data = np.repeat(unique_pixels, counts)
        sns.kdeplot(expanded_data, label=f"{temperature}°C")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    ax.set_xticks([0, 255])
    ax.set_xticklabels(['Black', 'White'])
    ax.set_title("Density Plot of Pixel Intensities by Temperature")
    plt.tight_layout()
    output_kde_file = f"{output_directory}/pixel_intensity_kde_{temperatures_str}.png"
    plt.savefig(output_kde_file, bbox_inches='tight')
    plt.close()
    print(f"Density plot saved to {output_kde_file}")


# Additional setups for seaborn
sns.set(style="whitegrid")
