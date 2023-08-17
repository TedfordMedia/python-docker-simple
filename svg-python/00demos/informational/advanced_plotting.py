# advanced_plotting.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def advanced_pixel_plotting(all_pixel_data, output_directory, temperatures_str):
    """
    Plot various visualizations for given pixel data.

    Args:
    - all_pixel_data (dict): Dictionary containing pixel data for images.
    - output_directory (str): Directory to save the plots.
    - temperatures_str (str): String representation of temperatures for naming.
    """

    # CDF of pixel intensities
    fig, ax = plt.subplots(figsize=(10, 6))
    for temperature, data in all_pixel_data.items():
        unique_pixels, counts, _ = data
        cdf = np.cumsum(counts)
        cdf = cdf / cdf[-1]
        ax.plot(unique_pixels, cdf, label=f"{temperature}°C", alpha=0.7)
    ax.legend()
    ax.set_xlabel("Gray Shade")
    ax.set_ylabel("CDF")
    ax.set_title("Pixel Intensity Cumulative Distribution by Temperature")
    plt.tight_layout()
    output_cdf_file = f"{output_directory}/pixel_intensity_cdf_{temperatures_str}.png"
    plt.savefig(output_cdf_file, bbox_inches='tight')
    plt.close()
    print(f"CDF graph saved to {output_cdf_file}")

    # Histogram of pixel intensities
    fig, ax = plt.subplots(figsize=(10, 6))
    for temperature, data in all_pixel_data.items():
        unique_pixels, counts, _ = data
        ax.bar(unique_pixels, counts,
               label=f"{temperature}°C", alpha=0.5, width=1)
    ax.legend()
    ax.set_xlabel("Gray Shade")
    ax.set_ylabel("Pixel Count")
    ax.set_title("Pixel Shade Distribution by Temperature")
    plt.tight_layout()
    output_hist_file = f"{output_directory}/pixel_intensity_histogram_{temperatures_str}.png"
    plt.savefig(output_hist_file, bbox_inches='tight')
    plt.close()
    print(f"Histogram graph saved to {output_hist_file}")

    # Violin plot (requires seaborn)
    all_data = []
    all_temps = []
    for temperature, data in all_pixel_data.items():
        unique_pixels, counts, _ = data
        # Expand to original data
        expanded_data = np.repeat(unique_pixels, counts)
        all_data.extend(expanded_data)
        all_temps.extend([temperature] * len(expanded_data))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x=all_temps, y=all_data)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Pixel Intensity")
    ax.set_title("Pixel Intensity Distribution by Temperature")
    plt.tight_layout()
    output_violin_file = f"{output_directory}/pixel_intensity_violin_{temperatures_str}.png"
    plt.savefig(output_violin_file, bbox_inches='tight')
    plt.close()
    print(f"Violin plot saved to {output_violin_file}")


# Additional setups for seaborn (if you choose to use the violin plot)
sns.set(style="whitegrid")
